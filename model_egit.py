"""
=============================================================================
ADIM 2: MODEL EĞİTİMİ - 4 Katmanlı LSTM + Holistic (150 Kare)
=============================================================================
Proje      : İşaret Dili Çevirmeni (TİD → Türkçe)
Öğrenci    : Fidan Zeynallı
Versiyon   : v3.0 (tid_holistic_model_v2.keras)
Giriş      : lstm_verisetim_v3.csv  (Adım 1'in çıktısı)
Çıkış      : tid_holistic_model_v2.keras + siniflar.npy
=============================================================================
MATRİS BOYUTU (Beklenen):
  X_train : (N, 150, 258)  ← 150 kare, 258 holistic koordinat
  y_train : (N, SINIF_SAYISI)  ← One-hot encoded
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# ─────────────────────────────────────────────
# PARAMETRELER  (Adım 1 ile tam senkron!)
# ─────────────────────────────────────────────
SEQUENCE_LENGTH = 60
OZELLIK_SAYISI  = 258
VERI_DOSYASI    = "lstm_verisetim_v4.csv"
MODEL_ADI       = "tid_holistic_model_v2.keras"
SINIFLAR_DOSYASI = "siniflar.npy"
TEST_ORANI      = 0.15   # %15 test, %85 eğitim
BATCH_SIZE      = 32
MAX_EPOCH       = 200    # Early Stopping erken bitirecek zaten

print("=" * 60)
print("🧠 4 Katmanlı LSTM Model Eğitimi Başlıyor")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. VERİYİ YÜKLE VE RESHAPE ET
# ─────────────────────────────────────────────
print(f"\n📂 '{VERI_DOSYASI}' yükleniyor...")
df = pd.read_csv(VERI_DOSYASI)
print(f"   Satır: {df.shape[0]:,}  |  Sütun: {df.shape[1]:,}")

etiketler = df["etiket"].values
koordinatlar = df.drop(columns=["etiket"]).values.astype(np.float32)

# (N, 150*258) → (N, 150, 258)
print(f"\n🔄 Matris reshape ediliyor: ({len(koordinatlar)}, {SEQUENCE_LENGTH * OZELLIK_SAYISI}) → ({len(koordinatlar)}, {SEQUENCE_LENGTH}, {OZELLIK_SAYISI})")
X = koordinatlar.reshape(-1, SEQUENCE_LENGTH, OZELLIK_SAYISI)
print(f"   ✅ X.shape = {X.shape}")

# ─────────────────────────────────────────────
# 2. ETİKET KODLAMA (Label Encoding → One-Hot)
# ─────────────────────────────────────────────
le = LabelEncoder()
y_sayisal = le.fit_transform(etiketler)
TOPLAM_SINIF = len(le.classes_)
y = to_categorical(y_sayisal, num_classes=TOPLAM_SINIF)

# Sınıf isimlerini kaydet (kamera_text.py'de kullanılacak)
np.save(SINIFLAR_DOSYASI, le.classes_)
print(f"\n🏷️  {TOPLAM_SINIF} sınıf tespit edildi → '{SINIFLAR_DOSYASI}' kaydedildi")
print(f"   İlk 5 sınıf: {le.classes_[:5].tolist()}")

# ─────────────────────────────────────────────
# 3. EĞİTİM / TEST AYIRIMI
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_ORANI, random_state=42
)
print(f"\n📊 Eğitim seti: {X_train.shape[0]:,} örnek")
print(f"   Test seti  : {X_test.shape[0]:,} örnek")

# ─────────────────────────────────────────────
# 4. MODEL MİMARİSİ (4 Katmanlı LSTM)
# ─────────────────────────────────────────────
print("\n🏗️  Model mimarisi oluşturuluyor...")

model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, OZELLIK_SAYISI)),

    # ── 1. LSTM Katmanı ── (Dizi döndürür → return_sequences=True)
    # 256 hücre: Geniş zaman aralığındaki ham hareket rotalarını öğrenir
    LSTM(256, return_sequences=True, name="lstm_1"),
    BatchNormalization(name="bn_1"),
    Dropout(0.3, name="dropout_1"),

    # ── 2. LSTM Katmanı ── (Dizi döndürür)
    # 128 hücre: Orta seviye el geçiş kalıplarını (pattern) kodlar
    LSTM(128, return_sequences=True, name="lstm_2"),
    BatchNormalization(name="bn_2"),
    Dropout(0.3, name="dropout_2"),

    # ── 3. LSTM Katmanı ── (Dizi döndürür)
    # 64 hücre: İnce motor detayları ve parmak ritmini yakalar
    LSTM(64, return_sequences=True, name="lstm_3"),
    BatchNormalization(name="bn_3"),
    Dropout(0.3, name="dropout_3"),

    # ── 4. LSTM Katmanı ── (Sadece son özet vektörü döndürür)
    # 32 hücre: Tüm zaman serisini tek bir anlam vektörüne sıkıştırır
    LSTM(32, return_sequences=False, name="lstm_4"),
    BatchNormalization(name="bn_4"),
    Dropout(0.3, name="dropout_4"),

    # ── Karar Katmanı ──
    Dense(128, activation='relu', name="dense_karar"),

    # ── Çıkış Katmanı ──
    Dense(TOPLAM_SINIF, activation='softmax', name="cikis")
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.summary()

# ─────────────────────────────────────────────
# 5. CALLBACK'LER (Akıllı Eğitim Denetimi)
# ─────────────────────────────────────────────
callbacks = [
    # En iyi modeli otomatik kaydet
    ModelCheckpoint(
        MODEL_ADI,
        monitor='val_categorical_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Doğrulama doğruluğu 15 tur boyunca iyileşmezse dur
    EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    # Öğrenme hızını otomatik düşür (plateau'da)
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
]

# ─────────────────────────────────────────────
# 6. EĞİTİM
# ─────────────────────────────────────────────
print(f"\n🚀 Eğitim başlıyor... (Max {MAX_EPOCH} tur, erken durdurma aktif)")
print("─" * 60)

tarih = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=MAX_EPOCH,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# 7. SONUÇ RAPORU
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("📈 EĞİTİM TAMAMLANDI")
print("=" * 60)

test_kayip, test_dogruluk = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Doğruluğu : %{test_dogruluk * 100:.2f}")
print(f"   Test Kaybı    : {test_kayip:.4f}")
print(f"   Model Kaydedildi: '{MODEL_ADI}'")
print(f"   Sınıflar Kaydedildi: '{SINIFLAR_DOSYASI}'")

en_iyi_tur = np.argmax(tarih.history['val_categorical_accuracy']) + 1
en_iyi_dogruluk = max(tarih.history['val_categorical_accuracy'])
print(f"\n🏆 En İyi Tur: {en_iyi_tur}  |  En İyi Val Doğruluk: %{en_iyi_dogruluk * 100:.2f}")
print("\n💡 Sonraki Adım: kamera_text.py ile canlı test!")
