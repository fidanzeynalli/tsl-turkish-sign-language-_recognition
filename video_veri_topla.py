# Faz 8.2: Artimli (Incremental) Veri Toplama ve Text Normalization
# Bu kod, sadece YENI eklenen videolari tespit eder, isler ve mevcut CSV dosyasinin altina ekler.

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

video_klasoru = "videolar"
csv_dosya_adi = "benim_verisetim.csv"

if not os.path.exists(video_klasoru):
    os.makedirs(video_klasoru)
    print(f"Lutfen egitim videolarini .mp4 formatinda '{video_klasoru}' klasorune koyun.")
    exit()

# --- ARTIMLI ISLEME (INCREMENTAL PROCESSING) KONTROLU ---
islenmis_etiketler = set()
if os.path.exists(csv_dosya_adi):
    try:
        mevcut_df = pd.read_csv(csv_dosya_adi)
        # Mevcut CSV'deki benzersiz kelimeleri (etiketleri) bir kume (set) icine al
        islenmis_etiketler = set(mevcut_df['etiket'].unique())
        print(f"-> Mevcut veri seti bulundu! Toplam {len(islenmis_etiketler)} kelime zaten islenmis durumda.")
    except Exception as e:
        print("CSV okunurken bir hata olustu:", e)

print(f"\n2. '{video_klasoru}' icindeki YENI videolar kontrol ediliyor...")

videolar = [v for v in os.listdir(video_klasoru) if v.endswith(".mp4") or v.endswith(".avi")]

if len(videolar) == 0:
    print("HATA: Klasorde hic video bulunamadi!")
    exit()

def turkce_karakter_temizle(metin):
    degisim_tablosu = str.maketrans("ğĞıİşŞöÖüÜçÇ", "gGiIsSoOuUcC")
    return metin.translate(degisim_tablosu).lower()

yeni_veriler = []

for video_adi in videolar:
    ham_etiket = video_adi.split('.')[0]
    etiket = turkce_karakter_temizle(ham_etiket)
    
    # Eger bu kelime CSV'de zaten varsa, hic yorulmadan es gec (Atla)
    if etiket in islenmis_etiketler:
        print(f" - Atlaniyor: '{video_adi}' (Zaten veri setinde mevcut)")
        continue
    
    video_yolu = os.path.join(video_klasoru, video_adi)
    cap = cv2.VideoCapture(video_yolu)
    kare_sayaci = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 
        
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
                
                # NORMALIZASYON
                bilek_x, bilek_y, bilek_z = noktalar[0], noktalar[1], noktalar[2]
                noktalar[0::3] -= bilek_x 
                noktalar[1::3] -= bilek_y 
                noktalar[2::3] -= bilek_z 

                if el_turu == 'Left':
                    frame_koordinatlari[:63] = noktalar
                else:
                    frame_koordinatlari[63:] = noktalar
                    
            satir_verisi = [etiket] + frame_koordinatlari.tolist()
            yeni_veriler.append(satir_verisi)

    cap.release()
    print(f" -> YENI EKLENDI: '{video_adi}' | Makine Etiketi: '{etiket}'")

detector.close()

# --- YENI VERILERI MEVCUT CSV'YE EKLEME (APPEND) ---
if len(yeni_veriler) > 0:
    print("\n3. Yeni veriler CSV dosyasina (Veri Setine) isleniyor...")
    sutunlar = ["etiket"]
    for el in ["sol", "sag"]:
        for i in range(21):
            sutunlar.extend([f"x_{el}_{i}", f"y_{el}_{i}", f"z_{el}_{i}"])

    df_yeni = pd.DataFrame(yeni_veriler, columns=sutunlar)
    
    # Eger CSV zaten varsa altina ekle (mode='a'), yoksa yeni olustur
    if os.path.exists(csv_dosya_adi):
        df_yeni.to_csv(csv_dosya_adi, mode='a', header=False, index=False)
        print(f"\n--- HARIKA! {len(df_yeni)} YENI KARE MEVCUT VERI SETINE EKLENDI ---")
    else:
        df_yeni.to_csv(csv_dosya_adi, index=False)
        print(f"\n--- MUTHIS! TOPLAM {len(df_yeni)} KARELIK KENDI VERI SETINIZ OLUSTURULDU ---")
else:
    print("\n--- Islenecek yeni video bulunamadi. Veri setiniz zaten en guncel halinde! ---")