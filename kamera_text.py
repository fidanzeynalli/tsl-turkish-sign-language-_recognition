# FİNAL V5: Özel Tasarım Anatomi İskeleti ve Bütüncül Çevirmen
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

print("1. Holistic LSTM Modeli ve Kelime Sözlüğü Yükleniyor...")
model = load_model("tid_holistic_model.keras")
siniflar = np.load('siniflar.npy', allow_pickle=True)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE, num_hands=2
)
pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=0.3 
)

hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)

BAGLANTILAR = [
    (0, 1), (1, 2), (2, 3), (3, 4),        
    (0, 5), (5, 6), (6, 7), (7, 8),        
    (9, 10), (10, 11), (11, 12),           
    (13, 14), (14, 15), (15, 16),          
    (0, 17), (17, 18), (18, 19), (19, 20), 
    (5, 9), (9, 13), (13, 17)              
]

SEQUENCE_LENGTH = 20
sekans_hafizasi = []
tahmin_hafizasi = []
olusturulan_cumle = []
son_eklenen_kelime = ""

def koordinatlari_cikar(mp_image):
    pose_res = pose_detector.detect(mp_image)
    hand_res = hand_detector.detect(mp_image)
    
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_res.pose_landmarks[0]]).flatten() if pose_res.pose_landmarks else np.zeros(33*4)
    lh, rh = np.zeros(21*3), np.zeros(21*3)
    
    el_var_mi = False
    if hand_res.hand_landmarks:
        el_var_mi = True
        for i, landmarks in enumerate(hand_res.hand_landmarks):
            turu = hand_res.handedness[i][0].category_name
            flat = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            if turu == 'Left': lh = flat
            else: rh = flat
            
    return np.concatenate([pose, lh, rh]), pose_res, hand_res, el_var_mi

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n--- SİSTEM HAZIR: İŞARET YAPMAYA BAŞLAYABİLİRSİN ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    h, w, _ = frame.shape
    dikey_genislik = int(h * (9/16)) 
    baslangic_x = (w // 2) - (dikey_genislik // 2)
    bitis_x = baslangic_x + dikey_genislik
    frame = frame[:, baslangic_x:bitis_x] 
    h_new, w_new, _ = frame.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    koordinatlar, pose_res, hand_res, el_var_mi = koordinatlari_cikar(mp_image)
    
    # 1. GÖRSELLEŞTİRME: ÖZEL ANATOMİ İSKELETİ (Kutucuklu Tasarım)
    if pose_res.pose_landmarks:
        lm_list = pose_res.pose_landmarks[0]
        
        # Noktaları piksele çeviren yardımcı fonksiyon
        def get_pt(idx):
            return (int(lm_list[idx].x * w_new), int(lm_list[idx].y * h_new))
            
        try:
            pt_nose = get_pt(0)   # Baş/Burun
            pt_l_sh = get_pt(11)  # Sol Omuz
            pt_r_sh = get_pt(12)  # Sağ Omuz
            pt_l_el = get_pt(13)  # Sol Dirsek
            pt_r_el = get_pt(14)  # Sağ Dirsek
            pt_l_wr = get_pt(15)  # Sol Bilek
            pt_r_wr = get_pt(16)  # Sağ Bilek
            pt_l_hip = get_pt(23) # Sol Kalça
            pt_r_hip = get_pt(24) # Sağ Kalça

            # Matematiksel Nokta Hesaplamaları
            pt_neck = ((pt_l_sh[0] + pt_r_sh[0]) // 2, (pt_l_sh[1] + pt_r_sh[1]) // 2) # Boyun (Omuzların ortası)
            pt_mid_hip = ((pt_l_hip[0] + pt_r_hip[0]) // 2, (pt_l_hip[1] + pt_r_hip[1]) // 2) # Kalça ortası
            pt_chest = (pt_neck[0], pt_neck[1] + int((pt_mid_hip[1] - pt_neck[1]) * 0.2)) # Göğüs (Boynun %20 altı)

            # --- SİYAH KEMİK ÇİZGİLERİ ---
            cv2.line(frame, pt_nose, pt_neck, (0, 0, 0), 3) # Baş -> Boyun
            cv2.line(frame, pt_neck, pt_chest, (0, 0, 0), 3) # Boyun -> Göğüs
            cv2.line(frame, pt_neck, pt_l_sh, (0, 0, 0), 3) # Boyun -> Sol Omuz
            cv2.line(frame, pt_neck, pt_r_sh, (0, 0, 0), 3) # Boyun -> Sağ Omuz
            cv2.line(frame, pt_l_sh, pt_l_el, (0, 0, 0), 3) # Sol Omuz -> Sol Dirsek
            cv2.line(frame, pt_l_el, pt_l_wr, (0, 0, 0), 3) # Sol Dirsek -> Sol Bilek
            cv2.line(frame, pt_r_sh, pt_r_el, (0, 0, 0), 3) # Sağ Omuz -> Sağ Dirsek
            cv2.line(frame, pt_r_el, pt_r_wr, (0, 0, 0), 3) # Sağ Dirsek -> Sağ Bilek
            cv2.line(frame, pt_chest, pt_mid_hip, (0, 0, 0), 3) # Omurga (Göğüs -> Kalça)

            # --- KUTUCUK ÇİZİM FONKSİYONU ---
            def draw_box(pt, color, size=10):
                cv2.rectangle(frame, (pt[0]-size, pt[1]-size), (pt[0]+size, pt[1]+size), color, -1)

            # --- TASARIM RENKLERİ ---
            draw_box(pt_nose, (255, 255, 255), 12) # Baş (Beyaz Büyük Kutu)
            draw_box(pt_neck, (0, 0, 255), 8)      # Boyun (Kırmızı Orta Kutu)
            draw_box(pt_chest, (0, 255, 255), 12)  # Göğüs (Sarı Büyük Kutu)
            
            # Turuncu Omurga Dikdörtgeni (Göğüsten Kalçaya)
            rect_w = 15
            cv2.rectangle(frame, (pt_chest[0]-rect_w, pt_chest[1]+15), (pt_chest[0]+rect_w, pt_mid_hip[1]), (0, 165, 255), -1)

            # Yeşil Eklemler (Omuzlar, Dirsekler, Bilekler)
            yesil_renk = (102, 255, 102)
            for pt in [pt_l_sh, pt_r_sh, pt_l_el, pt_r_el, pt_l_wr, pt_r_wr]:
                draw_box(pt, yesil_renk, 10)

        except Exception as e:
            pass # Noktalardan biri ekrandan çıkarsa çökmemesi için koruma

    # 2. GÖRSELLEŞTİRME: Eller (Renkli Kutular ve Çizgiler)
    if hand_res.hand_landmarks:
        for landmarks in hand_res.hand_landmarks:
            nokta_pikselleri = [(int(lm.x * w_new), int(lm.y * h_new)) for lm in landmarks]
            
            for bag in BAGLANTILAR:
                cv2.line(frame, nokta_pikselleri[bag[0]], nokta_pikselleri[bag[1]], (255, 255, 255), 2) 
                
            for i, (cx, cy) in enumerate(nokta_pikselleri):
                if i == 0: renk = (255, 255, 255) 
                elif i in [1, 2, 3, 4]: renk = (255, 0, 0) 
                elif i in [5, 6, 7, 8]: renk = (0, 255, 255) 
                elif i in [9, 10, 11, 12]: renk = (0, 255, 0) 
                elif i in [13, 14, 15, 16]: renk = (255, 0, 255) 
                else: renk = (0, 0, 255) 
                cv2.rectangle(frame, (cx-5, cy-5), (cx+5, cy+5), renk, cv2.FILLED)

    # --- TAHMİN MANTIĞI ---
    anlik_tahmin = "Bekleniyor..."
    if el_var_mi:
        sekans_hafizasi.append(koordinatlar)
        sekans_hafizasi = sekans_hafizasi[-SEQUENCE_LENGTH:]
        
        if len(sekans_hafizasi) == SEQUENCE_LENGTH:
            res = model.predict(np.expand_dims(sekans_hafizasi, axis=0), verbose=0)[0]
            indeks = np.argmax(res)
            guven = res[indeks]
            
            if guven > 0.85:
                kelime = siniflar[indeks]
                anlik_tahmin = f"{kelime} (%{int(guven*100)})"
                tahmin_hafizasi.append(kelime)
                tahmin_hafizasi = tahmin_hafizasi[-10:]
                
                en_cok = max(set(tahmin_hafizasi), key=tahmin_hafizasi.count)
                if en_cok != son_eklenen_kelime:
                    olusturulan_cumle.append(en_cok)
                    son_eklenen_kelime = en_cok
    else:
        sekans_hafizasi = []
        anlik_tahmin = "El Gorunmuyor"

    # UI Paneli
    cv2.rectangle(frame, (0,0), (w_new, 40), (245, 117, 16), -1)
    cv2.putText(frame, f"Tahmin: {anlik_tahmin}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(frame, (0, h_new-50), (w_new, h_new), (0, 0, 0), -1)
    gosterilecek_metin = " ".join(olusturulan_cumle[-5:]) 
    cv2.putText(frame, f"Cumle: {gosterilecek_metin}", (10, h_new-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imshow('TID Ozel Tasarim İskelet', frame)
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): 
        olusturulan_cumle = []
        son_eklenen_kelime = ""

cap.release()
cv2.destroyAllWindows()