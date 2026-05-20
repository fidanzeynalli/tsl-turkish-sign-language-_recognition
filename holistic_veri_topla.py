# Faz 14 (GÜNCEL): Modern Tasks API ile Holistic (Bütüncül) Veri Çıkarımı
import cv2
import mediapipe as mp# El ve vücut eklem noktalarını (iskelet) bulmak için.
import numpy as np# Koordinatları matematiksel dizilere (array) çevirmek için.
import pandas as pd# Verileri tablo yapısına (CSV) sokup kaydetmek için
import os
import urllib.request# MediaPipe'ın ihtiyaç duyduğu "beyin" dosyalarını internetten indirmek için

print("1. MediaPipe Tasks API Modelleri Hazırlanıyor...")


# Eğer bilgisayarda 'hand_landmarker.task' yoksa Google sunucularından otomatik indirir.
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

# 1. El Dedektörü Kurulumu 2 eli birden takip edecek ve %50 eminlikte çalışacak şekilde ayarladım
hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5
)
hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

# 2. Vücut (Pose) Dedektörü Kurulumu Omuz, kol ve yüz hatlarını çıkarmak için kurulur.
pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=0.5
)
pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)

SEQUENCE_LENGTH = 20  # LSTM için her hareket 20 karelik bir sekans olarak saklanır.
video_klasoru = "videolar"
csv_dosya_adi = "holistic_verisetim.csv"
islenenler_dosyasi = "holistic_islenenler.txt"# Aynı videoyu iki kez işlememek için kayıt tutar.

# Daha önce işlenmiş videoları hafızaya al.
islenmis_videolar = set()
if os.path.exists(islenenler_dosyasi):
    with open(islenenler_dosyasi, "r") as f:
        islenmis_videolar = set([line.strip() for line in f])
# Klasördeki videoları listele.
videolar = [v for v in os.listdir(video_klasoru) if v.endswith(".mp4") or v.endswith(".avi")]

# Türkçe karakterleri İngilizceye çevirenfonk
def turkce_karakter_temizle(metin):
    degisim_tablosu = str.maketrans("ğĞıİşŞöÖüÜçÇ", "gGiIsSoOuUcC")
    return metin.translate(degisim_tablosu).lower()

#ASIL KOORDİNAT ÇIKARMA FONKSİYONU ---
def koordinatlari_cikar(mp_image):
    # O karedeki pozu ve elleri aynı anda tespit et.r
    pose_result = pose_detector.detect(mp_image)
    hand_result = hand_detector.detect(mp_image)
    
    # 1.Pose  Koordinatları (33 x 4 = 132 değer)
    if pose_result.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_result.pose_landmarks[0]]).flatten()
    else:
        pose = np.zeros(33 * 4)# Vücut yoksa 132 tane sıfır basıyor
        
    # 2. Elkoordinatları (Sol:63, Sağ: 63)
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    
    if hand_result.hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
            el_turu = hand_result.handedness[hand_idx][0].category_name
            noktalar = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
            
            if el_turu == 'Left': lh = noktalar
            else: rh = noktalar
                
# Toplamda 132(Vücut) + 63(Sol) + 63(Sağ) = 258 koordinatlık dev diziyi birleşt
    return np.concatenate([pose, lh, rh])

#!!!esas döngü :videoları işleme ve koordinatları çıkarma kısmı
yeni_veriler = []
yeni_islenen_videolar = []

print("\n2. Videolardan Vücut + El Koordinatları Çıkarılıyor (Bu işlem biraz sürebilir)...")

for video_adi in videolar:
    if video_adi in islenmis_videolar: continue# Zaten işlendiyse atla.
        
    etiket = turkce_karakter_temizle(video_adi.split('.')[0])# Video adını kelime etiketi yap.
    cap = cv2.VideoCapture(os.path.join(video_klasoru, video_adi))
    kare_sayaci = 0
    video_kareleri = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 

        # Her 2 karede 1'ini alarak veri setini gereksiz kalabalıktan kurtarır.
        kare_sayaci += 1
        if kare_sayaci % 2 != 0: continue 
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        # O karenin koordinatlarını çıkar ve listeye ekle.
        frame_koordinatlari = koordinatlari_cikar(mp_image)
        video_kareleri.append(frame_koordinatlari)

    cap.release()

#SLIDING WINDOW) OLUŞTURMA  ----
    if len(video_kareleri) == 0: continue
    # Hareket kısa sürdüyse eksik kareleri 258 tane sıfır (Padding) ile doldurur.   
    if len(video_kareleri) < SEQUENCE_LENGTH:
        eksik = SEQUENCE_LENGTH - len(video_kareleri)
        sekans = video_kareleri + [np.zeros(258, dtype=np.float32) for _ in range(eksik)]
        yeni_veriler.append([etiket] + np.array(sekans).flatten().tolist())
        sekans_sayisi = 1
    else:
        # Uzun videolardan 20'şer karelik kayan pencereler üreterek veriyi çoğaltır.
        sekans_sayisi = len(video_kareleri) - SEQUENCE_LENGTH + 1
        for i in range(sekans_sayisi):
            sekans = video_kareleri[i : i + SEQUENCE_LENGTH]
            yeni_veriler.append([etiket] + np.array(sekans).flatten().tolist())
            
    yeni_islenen_videolar.append(video_adi)
    print(f" -> İşlendi: '{video_adi}' | Üretilen Sekans: {sekans_sayisi}")

# Modelleri kapat
hand_detector.close()
pose_detector.close()

# VERİLERİ KAYDETME ---
if len(yeni_veriler) > 0:
    print("\n3. Bütüncül Zaman Serisi Verileri CSV'ye Yazılıyor...")
    # Sütun isimlerini oluştur (1 etiket + 20 kare x 258 koordinat = 5161 sütun).
    sutunlar = ["etiket"] + [f"koordinat_{i}" for i in range(SEQUENCE_LENGTH * 258)]
    df_yeni = pd.DataFrame(yeni_veriler, columns=sutunlar)
    
    # CSV varsa altına ekle, yoksa yeni oluştur.
    if os.path.exists(csv_dosya_adi): 
        df_yeni.to_csv(csv_dosya_adi, mode='a', header=False, index=False)
    else: df_yeni.to_csv(csv_dosya_adi, index=False)

  # İşlenen videoların çetelesini güncelle.   
    with open(islenenler_dosyasi, "a") as f:
        for v in yeni_islenen_videolar: f.write(v + "\n")
            
    print(f"\n--- MÜTHİŞ! {len(yeni_veriler)} ADET HOLISTIC (VÜCUT+EL) VERİSİ EKLENDİ ---")