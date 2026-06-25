# Faz 8.2: Artimli (Incremental) Veri Toplama ve Text Normalization
# Bu kod, sadece YENI eklenen videolari tespit eder, isler ve mevcut CSV dosyasinin altina ekler.

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import urllib.request

print("1. Kutuphaneler ve MediaPipe hazirlaniyor...")

#MEDİAPİPE EL MODELİNİN İNDİRİLMESİ ---
task_path = 'hand_landmarker.task'
if not os.path.exists(task_path):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, task_path)

# MediaPipe Tasks API ayarları
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# El tespiti için güven oranlarını (%50) ayarlıyoruz.
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

# --- ARTIMLI İŞLEME (KONTROL) BLOĞU ---
# Eğer txt dosyan varsa bunu kullan, yoksa boş bir set oluştur
islenmis_videolar = set()
if os.path.exists("normalized_islenenler.txt"):
    with open("normalized_islenenler.txt", "r") as f:
        islenmis_videolar = set([line.strip() for line in f])

print(f"\n2. '{video_klasoru}' icindeki YENI videolar kontrol ediliyor...")

# Klasördeki videoları listele.
videolar = [v for v in os.listdir(video_klasoru) if v.endswith(".mp4") or v.endswith(".avi")]
if len(videolar) == 0:
    print("HATA: Klasorde hic video bulunamadi!")
    exit()

# Türkçe karakterleri temizleme (Örn: 'İ' -> 'i', 'ğ' -> 'g')
def turkce_karakter_temizle(metin):
    degisim_tablosu = str.maketrans("ğĞıİşŞöÖüÜçÇ", "gGiIsSoOuUcC")
    return metin.translate(degisim_tablosu).lower()

yeni_veriler = []

#VİDEO İŞLEME DÖNGÜSÜ ---
for video_adi in videolar:
    ham_etiket = video_adi.split('.')[0]
    etiket = turkce_karakter_temizle(ham_etiket)
    
    # Kelime zaten CSV'de varsa, bu videoyu işlemeden atla.
    if etiket in islenmis_videolar:
        print(f" - Atlaniyor: '{video_adi}' (Zaten veri setinde mevcut)")
        continue
    
    video_yolu = os.path.join(video_klasoru, video_adi)
    cap = cv2.VideoCapture(video_yolu)
    kare_sayaci = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 
        
        # Her 2 karede 1 alarak veri setini gereksiz büyümeden korur.
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
                
               # --- BİLEK MERKEZLİ NORMALİZASYON ---
                # Tüm noktaları bileğin koordinatından çıkararak (0,0,0) noktasına sabitler.
                bilek_x, bilek_y, bilek_z = noktalar[0], noktalar[1], noktalar[2]
                noktalar[0::3] -= bilek_x 
                noktalar[1::3] -= bilek_y 
                noktalar[2::3] -= bilek_z 

                if el_turu == 'Left':
                    frame_koordinatlari[:63] = noktalar
                else:
                    frame_koordinatlari[63:] = noktalar
             # Etiket ve 126 koordinatı birleştirerek tek satır oluşturur.       
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
    
    # Eger CSV zaten varsa altina ekle (mode='a'), yoksa yeni olustur mode='a' (append) ile mevcut dosyanın sonuna ekleme yapılır.
    if os.path.exists(csv_dosya_adi):
        df_yeni.to_csv(csv_dosya_adi, mode='a', header=False, index=False)
        print(f"\n--- HARIKA! {len(df_yeni)} YENI KARE MEVCUT VERI SETINE EKLENDI ---")
    else:
        df_yeni.to_csv(csv_dosya_adi, index=False)
        print(f"\n--- MUTHIS! TOPLAM {len(df_yeni)} KARELIK KENDI VERI SETINIZ OLUSTURULDU ---")
else:
    print("\n--- Islenecek yeni video bulunamadi. Veri setiniz zaten en guncel halinde! ---")