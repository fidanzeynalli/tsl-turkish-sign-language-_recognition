# Faz 14 (Normalized): Burun-Merkezli Holistic (Bütüncül) Veri Çıkarımı
# Bu kod, tüm vücut ve el koordinatlarını burun noktasına göre normalleştirir.
# Bu sayede model ezberden kurtulup vücut oranlarını öğrenir.

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import urllib.request

print("1. MediaPipe Tasks API Modelleri Hazırlanıyor...")

# Task dosyalarını Google'dan indir (Eğer yoksa)
hand_model_path = 'hand_landmarker.task'
if not os.path.exists(hand_model_path):
    print("-> El modeli indiriliyor...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", hand_model_path)

pose_model_path = 'pose_landmarker.task'
if not os.path.exists(pose_model_path):
    print("-> Vücut (Pose) modeli indiriliyor...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", pose_model_path)

# MediaPipe Tasks API Kurulumu
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 1. El Dedektörü Kurulumu
hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5
)
hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

# 2. Vücut (Pose) Dedektörü Kurulumu
pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=0.5
)
pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)

SEQUENCE_LENGTH = 20  
video_klasoru = "videolar"
csv_dosya_adi = "normalized_verisetim.csv" # YENİ CSV ADI
islenenler_dosyasi = "normalized_islenenler.txt"

islenmis_videolar = set()
if os.path.exists(islenenler_dosyasi):
    with open(islenenler_dosyasi, "r") as f:
        islenmis_videolar = set([line.strip() for line in f])

videolar = [v for v in os.listdir(video_klasoru) if v.endswith(".mp4") or v.endswith(".avi")]

def turkce_karakter_temizle(metin):
    degisim_tablosu = str.maketrans("ğĞıİşŞöÖüÜçÇ", "gGiIsSoOuUcC")
    return metin.translate(degisim_tablosu).lower()

def koordinatlari_cikar(mp_image):
    pose_result = pose_detector.detect(mp_image)
    hand_result = hand_detector.detect(mp_image)
    
    pt_nose = np.zeros(3) # Burun varsayılan olarak sıfır

    # 1. Pose (Vücut) Koordinatları (33 x 4 = 132 değer)
    if pose_result.pose_landmarks:
        pose_landmarks = pose_result.pose_landmarks[0]
        # Burun noktasını (index 0) referans olarak al
        pt_nose = np.array([pose_landmarks[0].x, pose_landmarks[0].y, pose_landmarks[0].z])
        
        # Tüm pose noktalarını buruna göre normalize et
        pose = np.array([
            [lm.x - pt_nose[0], lm.y - pt_nose[1], lm.z - pt_nose[2], lm.visibility]
            for lm in pose_landmarks
        ]).flatten()
    else:
        pose = np.zeros(33 * 4)
        
    # 2. El Koordinatları (Sol: 63, Sağ: 63)
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    
    if hand_result.hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
            el_turu = hand_result.handedness[hand_idx][0].category_name
            # Tüm el noktalarını buruna göre normalize et
            noktalar = np.array([
                [lm.x - pt_nose[0], lm.y - pt_nose[1], lm.z - pt_nose[2]]
                for lm in hand_landmarks
            ]).flatten()
            
            if el_turu == 'Left': lh = noktalar
            else: rh = noktalar
                
    # Hepsini birleştirip o meşhur 258 normalized koordinatlık diziyi oluştur
    return np.concatenate([pose, lh, rh])

yeni_veriler = []
yeni_islenen_videolar = []

print("\n2. Videolardan Normalize Edilmiş (Burun Merkezli) Koordinatlar Çıkarılıyor...")

for video_adi in videolar:
    if video_adi in islenmis_videolar: continue
        
    etiket = turkce_karakter_temizle(video_adi.split('.')[0])
    cap = cv2.VideoCapture(os.path.join(video_klasoru, video_adi))
    kare_sayaci = 0
    video_kareleri = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 
        
        kare_sayaci += 1
        # Veri artırmak için her kareyi alıyoruz (Her 2 karede 1 değil)
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        frame_koordinatlari = koordinatlari_cikar(mp_image)
        video_kareleri.append(frame_koordinatlari)

    cap.release()
    
    if len(video_kareleri) == 0: continue
        
    # Zaman Serisi (Sliding Window)
    if len(video_kareleri) < SEQUENCE_LENGTH:
        eksik = SEQUENCE_LENGTH - len(video_kareleri)
        sekans = video_kareleri + [np.zeros(258, dtype=np.float32) for _ in range(eksik)]
        yeni_veriler.append([etiket] + np.array(sekans).flatten().tolist())
        sekans_sayisi = 1
    else:
        sekans_sayisi = len(video_kareleri) - SEQUENCE_LENGTH + 1
        for i in range(sekans_sayisi):
            sekans = video_kareleri[i : i + SEQUENCE_LENGTH]
            yeni_veriler.append([etiket] + np.array(sekans).flatten().tolist())
            
    yeni_islenen_videolar.append(video_adi)
    print(f" -> İşlendi (Nrml): '{video_adi}' | Üretilen Sekans: {sekans_sayisi}")

# Modelleri kapat
hand_detector.close()
pose_detector.close()

if len(yeni_veriler) > 0:
    print("\n3. Normalize Veriler CSV'ye Yazılıyor...")
    sutunlar = ["etiket"] + [f"koordinat_{i}" for i in range(SEQUENCE_LENGTH * 258)]
    df_yeni = pd.DataFrame(yeni_veriler, columns=sutunlar)
    
    if os.path.exists(csv_dosya_adi): df_yeni.to_csv(csv_dosya_adi, mode='a', header=False, index=False)
    else: df_yeni.to_csv(csv_dosya_adi, index=False)
        
    with open(islenenler_dosyasi, "a") as f:
        for v in yeni_islenen_videolar: f.write(v + "\n")
            
    print(f"\n--- MÜTHİŞ! {len(yeni_veriler)} ADET NORMALIZE HOLISTIC VERİSİ EKLENDİ ---")