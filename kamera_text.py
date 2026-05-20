"""
=============================================================================
ADIM 3: CANLI KAMERA - FPS Sabitleyici + Tolerans Motoru + Çoğunluk Oyu
=============================================================================
Proje      : İşaret Dili Çevirmeni (TİD → Türkçe)
Öğrenci    : Fidan Zeynallı
Versiyon   : V8 Final (150 Kare / Holistic / Nose-Centric / Anatomik İskelet)
Model      : tid_holistic_model_v2.keras
Açıklama   :
  • 30 FPS Zaman Sabitleyici  → Kamera hızından bağımsız sabit veri akışı
  • Tolerans Motoru (12 kare) → Anlık el kaybolmalarında hafızayı sıfırlamaz
  • Çoğunluk Oyu (15 kare)   → Son 15 tahminde ≥10 kez geçen kelimeyi yazar
  • Anatomik İskelet          → Renkli parmak eklemleri + omuz/omurga hatları
  • Burun Merkezli Norm.      → Scale-invariant koordinat sistemi
=============================================================================
ÇALIŞTIRMAK İÇİN:
  pip install opencv-python mediapipe tensorflow numpy
  python kamera_text.py
  [Q] tuşuna bas → çıkış
=============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
from types import SimpleNamespace
from collections import Counter
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# 1. PARAMETRELER VE MODEL YÜKLEME
# ─────────────────────────────────────────────
SEQUENCE_LENGTH = 60       # Eğitim verisiyle tam senkron (150 kare = ~5 saniye @30FPS)
OZELLIK_SAYISI  = 258        # Burun merkezli holistic koordinat sayısı
GUVEN_ESIGI     = 0.85       # %85 güven altındaki tahminler "Emin Değil" sayılır
TOLERANS_ESIGI  = 12         # El anlık kaybolursa kaç kare sabır gösterelim? (~0.4 sn)
HAVUZ_BOYUTU    = 15         # Çoğunluk oyu için son kaç tahmin tutulsun?
KARARLILIK_ESIGI = 10        # Havuzda en az kaç kez aynı kelime çıkmalı?
HEDEF_FPS       = 30.0       # Eğitim videolarının FPS'i → sabitlemek zorundayız
HEDEF_SURE      = 1.0 / HEDEF_FPS   # 33.33 ms

print("🧠 4 Katmanlı LSTM Modeli yükleniyor...")
try:
    model_dosyasi = 'tid_holistic_model_v2.keras'
    if not os.path.exists(model_dosyasi):
        model_dosyasi = 'tid_holistic_model.keras'
    model   = load_model(model_dosyasi)
    siniflar = np.load('siniflar.npy', allow_pickle=True)
    print(f"✅ Model hazır. Sınıf sayısı: {len(siniflar)}")
except Exception as e:
    print(f"❌ Model yüklenemedi: {e}")
    print("   'tid_holistic_model_v2.keras' veya 'tid_holistic_model.keras' ve 'siniflar.npy' aynı klasörde olmalı!")
    exit(1)

# ─────────────────────────────────────────────
# 2. MEDİAPIPE TASKS KURULUMU
# ─────────────────────────────────────────────
hand_model_path = 'hand_landmarker.task'
pose_model_path = 'pose_landmarker.task'

if not os.path.exists(hand_model_path):
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        hand_model_path,
    )

if not os.path.exists(pose_model_path):
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
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
            if el_turu == 'Left':
                left_hand_landmarks = el_nesnesi
            else:
                right_hand_landmarks = el_nesnesi

    return SimpleNamespace(
        pose_landmarks=pose_landmarks,
        left_hand_landmarks=left_hand_landmarks,
        right_hand_landmarks=right_hand_landmarks,
    )

# ─────────────────────────────────────────────
# 3. BURUN MERKEZLİ NORMALİZASYON FONKSİYONU
# ─────────────────────────────────────────────
def burun_merkezli_normalize(ham_liste, bx, by, bz, olcek):
    """
    Ham koordinat listesini [x,y,z, x,y,z, ...] biçiminde alır,
    burun referans noktasına (bx,by,bz) göre ortalar ve ölçek ile normalize eder.
    """
    cikti = []
    for i in range(0, len(ham_liste), 3):
        cikti.append((ham_liste[i]   - bx) / (olcek + 1e-6))
        cikti.append((ham_liste[i+1] - by) / (olcek + 1e-6))
        cikti.append((ham_liste[i+2] - bz) / (olcek + 1e-6))
    return cikti


def kareden_vektor_cek(sonuclar):
    """
    MediaPipe Holistic sonuçlarından 258 boyutlu normalize vektör üretir.
    Sıra: Sol El (63) + Sağ El (63) + Pose Üst Vücut (99) + Padding (33)
    Döndürür: (np.array(258,), el_var_mi: bool)
    """
    if not sonuclar.pose_landmarks:
        return np.zeros(OZELLIK_SAYISI, dtype=np.float32), False

    pose = sonuclar.pose_landmarks.landmark

    # Burun referansı (Pose indeks 0)
    bx, by, bz = pose[0].x, pose[0].y, pose[0].z

    # Ölçek: iki omuz arası mesafe (indeks 11=sol omuz, 12=sağ omuz)
    olcek = np.sqrt(
        (pose[11].x - pose[12].x) ** 2 +
        (pose[11].y - pose[12].y) ** 2
    ) + 1e-6

    koordinatlar = []
    el_var = False

    # Sol el — 21 nokta × 3 = 63 koordinat
    if sonuclar.left_hand_landmarks:
        el_var = True
        ham = []
        for lm in sonuclar.left_hand_landmarks.landmark:
            ham.extend([lm.x, lm.y, lm.z])
        koordinatlar += burun_merkezli_normalize(ham, bx, by, bz, olcek)
    else:
        koordinatlar += [0.0] * 63

    # Sağ el — 21 nokta × 3 = 63 koordinat
    if sonuclar.right_hand_landmarks:
        el_var = True
        ham = []
        for lm in sonuclar.right_hand_landmarks.landmark:
            ham.extend([lm.x, lm.y, lm.z])
        koordinatlar += burun_merkezli_normalize(ham, bx, by, bz, olcek)
    else:
        koordinatlar += [0.0] * 63

    # Pose tüm vücut — 33 nokta × 3 = 99 koordinat
    ham = []
    for lm in pose:
        ham.extend([lm.x, lm.y, lm.z])
    koordinatlar += burun_merkezli_normalize(ham, bx, by, bz, olcek)

    # Toplam şimdilik: 63+63+99 = 225 — 258'e padding ile tamamla
    if len(koordinatlar) < OZELLIK_SAYISI:
        koordinatlar += [0.0] * (OZELLIK_SAYISI - len(koordinatlar))
    elif len(koordinatlar) > OZELLIK_SAYISI:
        koordinatlar = koordinatlar[:OZELLIK_SAYISI]

    return np.array(koordinatlar, dtype=np.float32), el_var


# ─────────────────────────────────────────────
# 4. ANATOMİK İSKELET ÇİZİM FONKSİYONLARI
# ─────────────────────────────────────────────

# Her parmak için renk (BGR)
PARMAK_RENKLERI = {
    "basparmak":   (0,   140, 255),   # Turuncu
    "isaret":      (0,   255, 100),   # Yeşil
    "orta":        (255, 200,   0),   # Sarı
    "yuzuk":       (200,   0, 255),   # Mor
    "serce":       (0,    80, 255),   # Kırmızı
    "avuc":        (255, 255, 255),   # Beyaz
}

# MediaPipe el landmark indeks grupları
EL_BAGLANTILARI = {
    "basparmak": [(0,1),(1,2),(2,3),(3,4)],
    "isaret":    [(0,5),(5,6),(6,7),(7,8)],
    "orta":      [(0,9),(9,10),(10,11),(11,12)],
    "yuzuk":     [(0,13),(13,14),(14,15),(15,16)],
    "serce":     [(0,17),(17,18),(18,19),(19,20)],
    "avuc":      [(5,9),(9,13),(13,17)],
}

# Pose iskeleti için bağlantı grupları (sol/sağ omuz–dirsek–bilek + omurga)
POSE_BAGLANTILARI = [
    (11, 12, (200, 200, 200)),   # Omuzlar arası — gri
    (11, 13, (0, 255, 200)),     # Sol üst kol   — turkuaz
    (13, 15, (0, 200, 255)),     # Sol ön kol    — açık mavi
    (12, 14, (0, 255, 200)),     # Sağ üst kol   — turkuaz
    (14, 16, (0, 200, 255)),     # Sağ ön kol    — açık mavi
    (11, 23, (180, 180, 180)),   # Sol omur hattı— açık gri
    (12, 24, (180, 180, 180)),   # Sağ omur hattı
    (23, 24, (160, 160, 160)),   # Kalça çizgisi
]

def el_iskeleti_ciz(kare, el_landmarks, h, w):
    """Tek bir el için renkli eklem kutuları ve parmak çizgileri çizer."""
    if el_landmarks is None:
        return
    noktalar = [(int(lm.x * w), int(lm.y * h)) for lm in el_landmarks.landmark]

    for parmak, baglar in EL_BAGLANTILARI.items():
        renk = PARMAK_RENKLERI[parmak]
        for (a, b) in baglar:
            cv2.line(kare, noktalar[a], noktalar[b], renk, 2, cv2.LINE_AA)

    # Eklem noktaları — küçük dolgu kareler
    for i, (px, py) in enumerate(noktalar):
        boyut = 5 if i == 0 else 3   # Bilek noktası biraz daha büyük
        cv2.rectangle(kare, (px-boyut, py-boyut), (px+boyut, py+boyut),
                      (255, 255, 255), -1)
        cv2.rectangle(kare, (px-boyut, py-boyut), (px+boyut, py+boyut),
                      (80, 80, 80), 1)


def pose_iskeleti_ciz(kare, pose_landmarks, h, w):
    """Omuz, omurga ve üst vücut iskelet çizgilerini çizer."""
    if pose_landmarks is None:
        return
    noktalar = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks.landmark]

    for (a, b, renk) in POSE_BAGLANTILARI:
        if a < len(noktalar) and b < len(noktalar):
            cv2.line(kare, noktalar[a], noktalar[b], renk, 2, cv2.LINE_AA)

    # Omuz ve kalça referans noktaları
    for idx in [11, 12, 23, 24]:
        if idx < len(noktalar):
            cv2.circle(kare, noktalar[idx], 6, (255, 255, 200), -1)
            cv2.circle(kare, noktalar[idx], 6, (100, 100, 100), 1)


# ─────────────────────────────────────────────
# 5. DURUM DEĞİŞKENLERİ
# ─────────────────────────────────────────────
sekans_hafizasi       = []    # LSTM'e beslenecek 150 karelik kayan pencere
anlik_tahmin_havuzu   = []    # Çoğunluk oyu için son 15 tahmini tutar
son_gecerli_koordinat = np.zeros(OZELLIK_SAYISI, dtype=np.float32)
bos_kare_sayaci       = 0
son_zaman             = time.time()

# Ekranda gösterilecek aktif kelime ve durum metni
aktif_kelime  = ""
durum_metni   = ""
aktif_renk    = (0, 255, 0)

# ─────────────────────────────────────────────
# 6. KAMERA BAŞLATMA
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Kamera açılamadı!")
    exit(1)

print("🎥 Kamera başlatıldı. İşaret dili çevirisi aktif!")
print("   [Q] tuşuna basarak çıkabilirsin.\n")

# ─────────────────────────────────────────────
# 7. ANA DÖNGÜ
# ─────────────────────────────────────────────
while cap.isOpened():

    # ── 7A. 30 FPS ZAMAN SABİTLEYİCİ ──────────────────────────────────────
    simdiki_zaman = time.time()
    gecen_sure    = simdiki_zaman - son_zaman

    if gecen_sure < HEDEF_SURE:
        # Hedef 33.3ms dolmadan kareyi atla → CPU meşgul etme
        ret, frame = cap.read()   # Kamerayı yine de oku ki buffer dolmasın
        if not ret:
            break
        continue

    son_zaman = simdiki_zaman

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ── 7B. MEDİAPIPE HOLISTIC İŞLEMİ ─────────────────────────────────────
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    pose_sonucu = pose_detector.detect(mp_image)
    hand_sonucu = hand_detector.detect(mp_image)
    sonuclar = tasks_sonucunu_holistic_formatina_cevir(pose_sonucu, hand_sonucu)

    # ── 7C. KOORDİNAT VEKTÖRÜ ÇIKARIMI ────────────────────────────────────
    current_landmarks, el_saptandi = kareden_vektor_cek(sonuclar)

    # ── 7D. TOLERANS MOTORU (FORWARD FILL) ─────────────────────────────────
    if not el_saptandi:
        bos_kare_sayaci += 1
        if bos_kare_sayaci < TOLERANS_ESIGI:
            # Anlık kopma: hafızayı sıfırlama, son koordinatı kopyala
            current_landmarks = son_gecerli_koordinat.copy()
        else:
            # Kullanıcı elini gerçekten indirdi → temiz başlangıç
            sekans_hafizasi.clear()
            anlik_tahmin_havuzu.clear()
            current_landmarks      = np.zeros(OZELLIK_SAYISI, dtype=np.float32)
            aktif_kelime           = ""
            durum_metni            = "El Bekleniyor..."
            aktif_renk             = (100, 100, 100)
    else:
        bos_kare_sayaci       = 0
        son_gecerli_koordinat = current_landmarks.copy()

    # ── 7E. 150 KARELİK KAYAN PENCERE ──────────────────────────────────────
    hafizaya_ekle = el_saptandi or (
        bos_kare_sayaci < TOLERANS_ESIGI and len(sekans_hafizasi) > 0
    )
    if hafizaya_ekle:
        sekans_hafizasi.append(current_landmarks)
        if len(sekans_hafizasi) > SEQUENCE_LENGTH:
            sekans_hafizasi.pop(0)   # En eski kareyi at → pencere kayar

    # ── 7F. TAHMİN + ÇOĞUNLUK OYU ──────────────────────────────────────────
    if len(sekans_hafizasi) == SEQUENCE_LENGTH:
        girdi = np.expand_dims(np.array(sekans_hafizasi), axis=0)  # (1, 150, 258)
        cikti = model.predict(girdi, verbose=0)[0]
        en_yuksek = np.argmax(cikti)
        guven     = float(cikti[en_yuksek])

        # Güven eşiğini geçmeyenler havuza -1 olarak girer
        anlik_tahmin_havuzu.append(en_yuksek if guven > GUVEN_ESIGI else -1)
        if len(anlik_tahmin_havuzu) > HAVUZ_BOYUTU:
            anlik_tahmin_havuzu.pop(0)

        gecerli = [t for t in anlik_tahmin_havuzu if t != -1]
        if gecerli:
            en_sik, sayi = Counter(gecerli).most_common(1)[0]
            if sayi >= KARARLILIK_ESIGI:
                aktif_kelime = siniflar[en_sik]
                durum_metni  = f"Tahmin: {aktif_kelime}  ({int(guven * 100)}%)"
                aktif_renk   = (0, 255, 80)
            else:
                durum_metni = "Hareket Analiz Ediliyor..."
                aktif_renk  = (0, 230, 255)
        else:
            durum_metni = "Sinyal Bekleniyor / Emin Degil"
            aktif_renk  = (0, 80, 255)
    else:
        durum_metni = (
            f"Sistem Isiniyor...  "
            f"({len(sekans_hafizasi)}/{SEQUENCE_LENGTH} kare)"
        )
        aktif_renk = (255, 140, 0)

    # ── 7G. ANATOMİK İSKELET ÇİZİMİ ───────────────────────────────────────
    pose_iskeleti_ciz(frame, sonuclar.pose_landmarks, h, w)
    el_iskeleti_ciz(frame,   sonuclar.left_hand_landmarks,  h, w)
    el_iskeleti_ciz(frame,   sonuclar.right_hand_landmarks, h, w)

    # ── 7H. EKRAN BİLGİ KATMANI (HUD) ──────────────────────────────────────

    # Üst yarı saydam şerit (durum metni arka planı)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Ana durum metni
    cv2.putText(
        frame, durum_metni,
        (30, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1,
        aktif_renk, 2, cv2.LINE_AA
    )

    # Sağ üst köşe: FPS ve hafıza göstergesi
    fps_anlık = 1.0 / max(gecen_sure, 1e-6)
    bilgi_str = (
        f"FPS: {fps_anlık:.0f}  |  "
        f"Hafiza: {min(len(sekans_hafizasi), SEQUENCE_LENGTH)}/{SEQUENCE_LENGTH}  |  "
        f"Tolerans: {bos_kare_sayaci}/{TOLERANS_ESIGI}"
    )
    cv2.putText(
        frame, bilgi_str,
        (30, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (160, 160, 160), 1, cv2.LINE_AA
    )

    # Hafıza dolum çubuğu (sol kenar)
    dolum_orani  = len(sekans_hafizasi) / SEQUENCE_LENGTH
    cubuk_yuksek = int((h - 100) * dolum_orani)
    cv2.rectangle(frame, (8, h - 90), (22, 90),            (60, 60, 60),  -1)
    cv2.rectangle(frame, (8, h - 90 - cubuk_yuksek + (h - 180)),
                          (22, h - 90),
                          (0, 200, 120), -1)

    # ── 7I. GÖRÜNTÜYÜ GÖSTER ───────────────────────────────────────────────
    cv2.imshow('TID → Türkçe | Gerçek Zamanlı İşaret Dili Çevirisi', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─────────────────────────────────────────────
# 8. TEMİZLİK
# ─────────────────────────────────────────────
hand_detector.close()
pose_detector.close()
cap.release()
cv2.destroyAllWindows()
print("\n👋 Sistem kapatıldı.")
