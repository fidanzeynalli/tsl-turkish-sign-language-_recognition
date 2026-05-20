"""
=============================================================================
ADIM 1: VERİ HAZIRLAMA - 60 Karelik Kayan Pencere + Burun Merkezli Normalizasyon
=============================================================================
Proje      : İşaret Dili Çevirmeni (TİD → Türkçe)
Öğrenci    : Fidan Zeynallı
Versiyon   : v4.0 (60 Kare / Holistic Tasks / Nose-Centric / Canlı İlerleme Takipçisi)
Açıklama   : Her videoyu 60 karelik paketlere böler. Çıktı matrisinin boyutu:
             (Toplam Örnek Sayısı, 60, 258)  <-- LSTM'e hazır kompakt veri
=============================================================================
KLASÖR YAPISI:
  videolar/
    Boşanmak.mp4
    Ceza.mp4
    ...
=============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import urllib.request
import time
from types import SimpleNamespace

# ─────────────────────────────────────────────
# PARAMETRELER (60 Kare Standardı)
# ─────────────────────────────────────────────
SEQUENCE_LENGTH = 60     # Her örnek tam 60 kare içerecek (~2 saniye)
ADIM_BOYUTU    = 15     # Kayan pencere adım büyüklüğü
OZELLIK_SAYISI = 258     # Burun merkezli holistic koordinat sayısı
VIDEO_KLASORU  = "videolar" 
CIKTI_DOSYASI  = "lstm_verisetim_v4.csv"

# ─────────────────────────────────────────────
# MEDİAPIPE TASKS MODEL YÜKLEME VE KURULUMU
# ─────────────────────────────────────────────
hand_model_path = "hand_landmarker.task"
pose_model_path = "pose_landmarker.task"

if not os.path.exists(hand_model_path):
    print("📥 Hand Landmarker modeli indiriliyor...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        hand_model_path,
    )

if not os.path.exists(pose_model_path):
    print("📥 Pose Landmarker modeli indiriliyor...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        pose_model_path,
    )

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)
hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
)
pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)

# ─────────────────────────────────────────────
# YARDIMCI DÖNÜŞTÜRME VE NORMALİZASYON FONKSİYONLARI
# ─────────────────────────────────────────────
def tasks_sonucunu_holistic_formatina_cevir(pose_sonucu, hand_sonucu):
    pose_landmarks = None
    left_hand_landmarks = None
    right_hand_landmarks = None

    if pose_sonucu.pose_landmarks:
        pose_landmarks = SimpleNamespace(landmark=pose_sonucu.pose_landmarks[0])

    if hand_sonucu.hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hand_sonucu.hand_landmarks):
            el_turu = hand_sonucu.handedness[hand_idx][0].category_name
            el_nesnesi = SimpleNamespace(landmark=hand_landmarks)
            if el_turu == "Left":
                left_hand_landmarks = el_nesnesi
            else:
                right_hand_landmarks = el_nesnesi

    return SimpleNamespace(
        pose_landmarks=pose_landmarks,
        left_hand_landmarks=left_hand_landmarks,
        right_hand_landmarks=right_hand_landmarks,
    )

def burun_merkezli_normalize_et(landmarks_ham, burun_x, burun_y, burun_z, olcek):
    normalize_edilmis = []
    for i in range(0, len(landmarks_ham), 3):
        nx = (landmarks_ham[i]   - burun_x) / (olcek + 1e-6)
        ny = (landmarks_ham[i+1] - burun_y) / (olcek + 1e-6)
        nz = (landmarks_ham[i+2] - burun_z) / (olcek + 1e-6)
        normalize_edilmis.extend([nx, ny, nz])
    return normalize_edilmis

def kareden_koordinat_cek(sonuclar):
    if not sonuclar.pose_landmarks:
        return None

    pose_pts = sonuclar.pose_landmarks.landmark
    burun_x = pose_pts[0].x
    burun_y = pose_pts[0].y
    burun_z = pose_pts[0].z

    olcek = np.sqrt((pose_pts[11].x - pose_pts[12].x)**2 + (pose_pts[11].y - pose_pts[12].y)**2)
    koordinatlar = []

    # 1) Sol el (63 koordinat)
    if sonuclar.left_hand_landmarks:
        sol_ham = []
        for lm in sonuclar.left_hand_landmarks.landmark:
            sol_ham.extend([lm.x, lm.y, lm.z])
        koordinatlar += burun_merkezli_normalize_et(sol_ham, burun_x, burun_y, burun_z, olcek)
    else:
        koordinatlar += [0.0] * 63

    # 2) Sağ el (63 koordinat)
    if sonuclar.right_hand_landmarks:
        sag_ham = []
        for lm in sonuclar.right_hand_landmarks.landmark:
            sag_ham.extend([lm.x, lm.y, lm.z])
        koordinatlar += burun_merkezli_normalize_et(sag_ham, burun_x, burun_y, burun_z, olcek)
    else:
        koordinatlar += [0.0] * 63

    # 3) Üst vücut pose (99 koordinat)
    pose_ham = []
    for lm in pose_pts:
        pose_ham.extend([lm.x, lm.y, lm.z])
    koordinatlar += burun_merkezli_normalize_et(pose_ham, burun_x, burun_y, burun_z, olcek)

    # Vektör boyutu sabitleme (258 özellik)
    if len(koordinatlar) > OZELLIK_SAYISI:
        koordinatlar = koordinatlar[:OZELLIK_SAYISI]
    elif len(koordinatlar) < OZELLIK_SAYISI:
        koordinatlar += [0.0] * (OZELLIK_SAYISI - len(koordinatlar))

    return np.array(koordinatlar, dtype=np.float32)

def videodan_kare_dizisi_cek(video_yolu):
    cap = cv2.VideoCapture(video_yolu)
    kare_listesi = []
    son_gecerli = np.zeros(OZELLIK_SAYISI, dtype=np.float32)

    while cap.isOpened():
        ret, kare = cap.read()
        if not ret:
            break

        kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=kare_rgb)

        pose_sonucu = pose_detector.detect(mp_image)
        hand_sonucu = hand_detector.detect(mp_image)
        sonuclar = tasks_sonucunu_holistic_formatina_cevir(pose_sonucu, hand_sonucu)
        koordinat = kareden_koordinat_cek(sonuclar)

        if koordinat is not None:
            son_gecerli = koordinat
        kare_listesi.append(son_gecerli.copy())

    cap.release()
    return np.array(kare_listesi, dtype=np.float32)

def kayan_pencere_uygula(kare_dizisi, etiket):
    """
    Bir video kare dizisini 60 karelik örtüşen pencerelere böler (Sliding Window).
    Kısa videolar son kareyle 60'a tamamlanır (Padding).
    """
    ornekler = []
    n = len(kare_dizisi)

    if n < SEQUENCE_LENGTH:
        eksik = SEQUENCE_LENGTH - n
        son_kare = kare_dizisi[-1] if n > 0 else np.zeros(OZELLIK_SAYISI)
        dolgu    = np.tile(son_kare, (eksik, 1))
        kare_dizisi = np.vstack([kare_dizisi, dolgu])
        n = SEQUENCE_LENGTH

    for baslangic in range(0, n - SEQUENCE_LENGTH + 1, ADIM_BOYUTU):
        pencere = kare_dizisi[baslangic : baslangic + SEQUENCE_LENGTH]  # (60, 258)
        ornekler.append((pencere, etiket))

    return ornekler

# ─────────────────────────────────────────────
# ANA ÇALIŞTIRMA DÖNGÜSÜ
# ─────────────────────────────────────────────
if __name__ == "__main__":
    tum_ornekler = []
    sinif_sayaclari = {}

    if not os.path.exists(VIDEO_KLASORU):
        print(f"❌ '{VIDEO_KLASORU}' klasörü bulunamadı!")
        exit(1)

    entries = sorted(os.listdir(VIDEO_KLASORU))
    class_folders = [e for e in entries if os.path.isdir(os.path.join(VIDEO_KLASORU, e))]

    print(f"🧠 MediaPipe Beyni Aktif Edildi. İşlemler Başlıyor...")
    print(f"⚙️  Parametreler: SEQUENCE_LENGTH={SEQUENCE_LENGTH}, ADIM={ADIM_BOYUTU}, ÖZELLIK={OZELLIK_SAYISI}")
    print("─" * 60)

    if class_folders:
        print(f"📂 {len(class_folders)} kelime sınıfı (klasör) bulundu.")
        for k_idx, kelime in enumerate(class_folders):
            kelime_yolu = os.path.join(VIDEO_KLASORU, kelime)
            videolar = [f for f in os.listdir(kelime_yolu) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if not videolar:
                continue

            kelime_ornek_sayisi = 0
            for v_idx, video_dosyasi in enumerate(videolar):
                video_tam_yol = os.path.join(kelime_yolu, video_dosyasi)
                
                # Klasörlü yapı için Canlı İlerleme Göstergesi
                print(f"⏳ Klasör: [{k_idx+1}/{len(class_folders)}] | Video: [{v_idx+1}/{len(videolar)}] -> '{video_dosyasi}' işleniyor...")
                
                try:
                    kare_dizisi = videodan_kare_dizisi_cek(video_tam_yol)
                    if len(kare_dizisi) == 0:
                        print(f"  ⚠️  {video_dosyasi}: Koordinat yok, atlandı.")
                        continue
                    pencereler = kayan_pencere_uygula(kare_dizisi, kelime)
                    tum_ornekler.extend(pencereler)
                    kelime_ornek_sayisi += len(pencereler)
                except Exception as e:
                    print(f"  ❌ Hata [{video_dosyasi}]: {e}")

            sinif_sayaclari[kelime] = kelime_ornek_sayisi
            print(f"  ✅ '{kelime}' klasörü bitti! Toplam {kelime_ornek_sayisi} örnek üretildi.\n")
    else:
        # DÜZ DİZİN YAPISI (Senin Şu An Çalıştırdığın Bölüm)
        videolar = [f for f in entries if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        print(f"📂 {len(videolar)} video dosyası bulundu (düz dizin). Her dosya bir etiket olarak işlenecek.")
        print("─" * 60)
        
        for idx, video_dosyasi in enumerate(videolar):
            video_tam_yol = os.path.join(VIDEO_KLASORU, video_dosyasi)
            ham_etiket = os.path.splitext(video_dosyasi)[0]
            etiket = ham_etiket.replace(' ', '_')
            
            # 🔥 SUÇLUYU YAKALAYACAK OLAN EN KRİTİK LOG SATIRI:
            print(f"⏳ [{idx+1}/{len(videolar)}] '{video_dosyasi}' işlenmeye başlandı, lütfen bekleyin...")
            
            try:
                kare_dizisi = videodan_kare_dizisi_cek(video_tam_yol)
                if len(kare_dizisi) == 0:
                    print(f"  ⚠️  {video_dosyasi}: Koordinat çıkarılamadı, atlanıyor.")
                    continue
                pencereler = kayan_pencere_uygula(kare_dizisi, etiket)
                tum_ornekler.extend(pencereler)
                sinif_sayaclari[etiket] = sinif_sayaclari.get(etiket, 0) + len(pencereler)
                
                print(f"  ✅ '{video_dosyasi}' başarıyla bitti! -> {len(pencereler)} örnek üretildi.")
            except Exception as e:
                print(f"  ❌ '{video_dosyasi}' işlenirken hata: {e}")

    print("─" * 60)
    print(f"📊 TOPLAM ÜRETİLEN ÖRNEK SAYISI: {len(tum_ornekler)}")

    hand_detector.close()
    pose_detector.close()

    if len(tum_ornekler) == 0:
        print("❌ Hiç örnek üretilemedi!")
        exit(1)

    # ─────────────────────────────────────────────
    # CSV'ye KAYDET (60 Kare Standart Matrisi)
    # ─────────────────────────────────────────────
    print("💾 CSV dosyasına yazılıyor... (Lütfen bekleyin)")

    satirlar = []
    for pencere, etiket in tum_ornekler:
        duzlesmis = pencere.flatten().tolist() # (60 * 258 = 15480 özellik)
        satirlar.append([etiket] + duzlesmis)

    sutun_isimleri = ["etiket"] + [f"coord_{i}" for i in range(SEQUENCE_LENGTH * OZELLIK_SAYISI)]
    df = pd.DataFrame(satirlar, columns=sutun_isimleri)
    df.to_csv(CIKTI_DOSYASI, index=False)

    print(f"✅ '{CIKTI_DOSYASI}' başarıyla kaydedildi!")
    print(f"📐 Nihai Matris Boyutu: ({len(tum_ornekler)}, {SEQUENCE_LENGTH}, {OZELLIK_SAYISI})")