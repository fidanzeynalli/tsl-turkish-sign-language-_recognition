# Faz 12: LSTM Mimarisini Icin Zaman Serisi (Sequence) Veri Cikarimi
# Bu kod, videolari tek tek kareler halinde degil, 20 karelik (2520 koordinatli) 
# anlamli hareket rotalari (sekanslar) halinde LSTM modeline hazirlar.

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import urllib.request

print("1. Kutuphaneler ve MediaPipe hazirlaniyor...")

task_path = 'hand_landmarker.task'
if not os.path.exists(task_path):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, task_path)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=task_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = HandLandmarker.create_from_options(options)

# LSTM GEREKSINIMI: Her kelime icin 20 karelik bir "hafiza" (yaklasik 1 saniyelik hareket)
SEQUENCE_LENGTH = 20 
video_klasoru = "videolar"

# Yeni verileri eski yapidan ayirmak icin farkli isimler kullaniyoruz
csv_dosya_adi = "lstm_verisetim.csv"
islenenler_dosyasi = "lstm_islenen_videolar.txt"

if not os.path.exists(video_klasoru):
    os.makedirs(video_klasoru)
    print(f"Lutfen egitim videolarini '{video_klasoru}' klasorune koyun.")
    exit()

# --- ARTIMLI ISLEME (Hangi videolarin yapildigini isminden takip et) ---
islenmis_videolar = set()
if os.path.exists(islenenler_dosyasi):
    with open(islenenler_dosyasi, "r") as f:
        islenmis_videolar = set([line.strip() for line in f])
    print(f"-> Sistemde {len(islenmis_videolar)} adet LSTM formatinda islenmis video kaydi bulundu.")

videolar = [v for v in os.listdir(video_klasoru) if v.endswith(".mp4") or v.endswith(".avi")]

def turkce_karakter_temizle(metin):
    degisim_tablosu = str.maketrans("ğĞıİşŞöÖüÜçÇ", "gGiIsSoOuUcC")
    return metin.translate(degisim_tablosu).lower()

yeni_veriler = []
yeni_islenen_videolar = []

print("\n2. Videolardan LSTM Zaman Serisi (Sequence) cikariliyor. Bu islem Sliding Window sebebiyle biraz surebilir...")

for video_adi in videolar:
    if video_adi in islenmis_videolar:
        #print(f" - Atlaniyor: '{video_adi}'") # Terminali cok kirletmemesi icin gizlendi
        continue
        
    ham_etiket = video_adi.split('.')[0]
    etiket = turkce_karakter_temizle(ham_etiket)
    
    video_yolu = os.path.join(video_klasoru, video_adi)
    cap = cv2.VideoCapture(video_yolu)
    kare_sayaci = 0
    video_kareleri = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 
        
        # Gereksiz kopyalari silmek icin her 2 karede 1 aliyoruz
        kare_sayaci += 1
        if kare_sayaci % 2 != 0: continue
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)
        
        frame_koordinatlari = np.zeros(126, dtype=np.float32)

        if detection_result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                el_turu = detection_result.handedness[hand_idx][0].category_name
                noktalar = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32).flatten()
                
                bilek_x, bilek_y, bilek_z = noktalar[0], noktalar[1], noktalar[2]
                noktalar[0::3] -= bilek_x 
                noktalar[1::3] -= bilek_y 
                noktalar[2::3] -= bilek_z 

                if el_turu == 'Left':
                    frame_koordinatlari[:63] = noktalar
                else:
                    frame_koordinatlari[63:] = noktalar
                    
        video_kareleri.append(frame_koordinatlari)

    cap.release()
    
    # --- ZAMAN SERISI (SLIDING WINDOW) OLUSTURMA ---
    if len(video_kareleri) == 0:
        continue
        
    # Eger video 20 kareden kisaysa (cok hizli bir hareketse) sonunu sifirlarla doldur
    if len(video_kareleri) < SEQUENCE_LENGTH:
        eksik = SEQUENCE_LENGTH - len(video_kareleri)
        sifirlar = [np.zeros(126, dtype=np.float32) for _ in range(eksik)]
        sekans = video_kareleri + sifirlar
        
        satir = [etiket] + np.array(sekans).flatten().tolist()
        yeni_veriler.append(satir)
        sekans_sayisi = 1
    else:
        # Video uzunsa, kayan pencere ile ayni videodan onlarca farkli paket uret
        sekans_sayisi = len(video_kareleri) - SEQUENCE_LENGTH + 1
        for i in range(sekans_sayisi):
            sekans = video_kareleri[i : i + SEQUENCE_LENGTH]
            satir = [etiket] + np.array(sekans).flatten().tolist()
            yeni_veriler.append(satir)
            
    yeni_islenen_videolar.append(video_adi)
    print(f" -> Işlendi: '{video_adi}' | Uretilen Sekans (LSTM): {sekans_sayisi}")

detector.close()

# --- DEVASA CSV'YE KAYDETME ---
if len(yeni_veriler) > 0:
    print("\n3. Yeni zaman serisi verileri LSTM CSV'sine yaziliyor (Sutun Sayisi: 2521)...")
    
    # Sutun basliklarini dinamik olustur (1 etiket + 20 kare * 126 koordinat = 2521 sutun)
    sutunlar = ["etiket"]
    for k in range(SEQUENCE_LENGTH):
        for el in ["sol", "sag"]:
            for i in range(21):
                sutunlar.extend([f"kare{k}_{el}_{i}_x", f"kare{k}_{el}_{i}_y", f"kare{k}_{el}_{i}_z"])

    df_yeni = pd.DataFrame(yeni_veriler, columns=sutunlar)
    
    if os.path.exists(csv_dosya_adi):
        df_yeni.to_csv(csv_dosya_adi, mode='a', header=False, index=False)
    else:
        df_yeni.to_csv(csv_dosya_adi, index=False)
        
    # Hangi videolari isledigimizi text dosyasina kaydet
    with open(islenenler_dosyasi, "a") as f:
        for v in yeni_islenen_videolar:
            f.write(v + "\n")
            
    print(f"\n--- HARIKA! {len(yeni_veriler)} ADET ZAMAN SERISI 'lstm_verisetim.csv' DOSYASINA EKLENDI ---")
else:
    print("\n--- Islenecek yeni video bulunamadi. ---")