import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

print("1. MediaPipe Tasks API Modelleri Hazırlanıyor...")

hand_model_path = 'hand_landmarker.task'
pose_model_path = 'pose_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path), running_mode=VisionRunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.5)
hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path), running_mode=VisionRunningMode.IMAGE, min_pose_detection_confidence=0.5)
pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)

# YENİ: VİDEOLARIN ORTALAMA SÜRESİNE ODAKLANAN DOĞAL PENCERE (20 KARE)
SEQUENCE_LENGTH = 20  
video_klasoru = "videolar"
csv_dosya_adi = "benim_verisetim.csv"
islenenler_dosyasi = "normalized_islenenler.txt"

islenmis_videolar = set()
if os.path.exists(islenenler_dosyasi):
    with open(islenenler_dosyasi, "r") as f:
        islenmis_videolar = set([line.strip() for line in f])

videolar = [v for v in os.listdir(video_klasoru) if v.endswith(".mp4") or v.endswith(".avi")]

def turkce_karakter_temizle(metin):
    degisim_tablosu = str.maketrans("ğĞıİşŞöÖüÜçÇ", "gGiIsSoOuUcC")
    return metin.translate(degisim_tablosu).lower()

# SENİN YÜKLEDİĞİN KODDAKİ NORMALİZASYON MANTIĞI AYNEN KORUNDU
def koordinatlari_cikar(mp_image):
    pose_res = pose_detector.detect(mp_image)
    hand_res = hand_detector.detect(mp_image)
    
    pose = np.zeros(33 * 4)
    burun_x, burun_y, burun_z = 0, 0, 0
    
    if pose_res.pose_landmarks:
        lm_list = pose_res.pose_landmarks[0]
        burun_x, burun_y, burun_z = lm_list[0].x, lm_list[0].y, lm_list[0].z
        pose = np.array([[lm.x - burun_x, lm.y - burun_y, lm.z - burun_z, lm.visibility] for lm in lm_list]).flatten()

    lh, rh = np.zeros(21 * 3), np.zeros(21 * 3)
    el_var_mi = False # YENİ: Hayalet kare için kontrol eklendi
    
    if hand_res.hand_landmarks:
        el_var_mi = True
        for i, landmarks in enumerate(hand_res.hand_landmarks):
            turu = hand_res.handedness[i][0].category_name
            flat = np.array([[lm.x - burun_x, lm.y - burun_y, lm.z - burun_z] for lm in landmarks]).flatten()
            if turu == 'Left': lh = flat
            else: rh = flat
            
    return np.concatenate([pose, lh, rh]), el_var_mi

yeni_veriler = []
yeni_islenen_videolar = []

print("\n2. Videolardan Normalize Koordinatlar Çıkarılıyor (FPS Düzenlemesiyle)...")

for video_adi in videolar:
    if video_adi in islenmis_videolar: continue
        
    etiket = turkce_karakter_temizle(video_adi.split('.')[0])
    cap = cv2.VideoCapture(os.path.join(video_klasoru, video_adi))
    video_kareleri = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        if 'son_bilinen_koordinat' not in locals():
            son_bilinen_koordinat = np.zeros(258)

        koordinatlar, el_var_mi = koordinatlari_cikar(mp_image)
        
        # YENİ: Hayalet Kare (Flickering) Koruması (Kayıtlarda kopma olmasın diye)
        if el_var_mi:
            son_bilinen_koordinat = koordinatlar
        else:
            if np.sum(son_bilinen_koordinat) != 0:
                koordinatlar = son_bilinen_koordinat
                
        video_kareleri.append(koordinatlar)

    cap.release()
    if len(video_kareleri) == 0: continue
        
    # YENİ: DOĞAL HIZI BOZMADAN 20 KARELİK PENCERELER ÇIKARMA
    sekans_sayisi = 0
    if len(video_kareleri) < SEQUENCE_LENGTH:
        # Video 20 kareden kısaysa: Son duruşu kopyalayarak 20'ye tamamla (Zero Padding yerine Padding with Last Frame)
        eksik = SEQUENCE_LENGTH - len(video_kareleri)
        son_kare = video_kareleri[-1]
        sekans = video_kareleri + [son_kare for _ in range(eksik)]
        yeni_veriler.append([etiket] + np.array(sekans).flatten().tolist())
        sekans_sayisi = 1
    else:
        # Video 20 kareden uzunsa: Doğal hızında (frame atlamadan) peş peşe 20'lik pencereler oluştur (Sliding Window)
        sekans_sayisi = len(video_kareleri) - SEQUENCE_LENGTH + 1
        for i in range(sekans_sayisi):
            sekans = video_kareleri[i : i + SEQUENCE_LENGTH]
            yeni_veriler.append([etiket] + np.array(sekans).flatten().tolist())
            
    yeni_islenen_videolar.append(video_adi)
    print(f" -> İşlendi: '{video_adi}' | Üretilen 20-Karelik Sekans: {sekans_sayisi}")

hand_detector.close()
pose_detector.close()

if len(yeni_veriler) > 0:
    print("\n3. Normalize Veriler CSV'ye Yazılıyor...")
    sutunlar = ["etiket"] + [f"koordinat_{i}" for i in range(SEQUENCE_LENGTH * 258)]
    df_yeni = pd.DataFrame(yeni_veriler, columns=sutunlar)
    
    if os.path.exists(csv_dosya_adi): 
        df_yeni.to_csv(csv_dosya_adi, mode='a', header=False, index=False)
    else: 
        df_yeni.to_csv(csv_dosya_adi, index=False)
        
    with open(islenenler_dosyasi, "a") as f:
        for v in yeni_islenen_videolar: f.write(v + "\n")
            
    print(f"\n--- MÜTHİŞ! {len(yeni_veriler)} ADET NORMALIZE (BURUN MERKEZLİ) VERİ EKLENDİ ---")
else:
    print("\n--- Yeni islenecek video bulunamadı. ---")