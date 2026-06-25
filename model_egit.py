# Faz 15: Holistic (Bütüncül) LSTM Model Eğitimi
from pathlib import Path

# Faz 15: Holistic (Bütüncül) LSTM Model Eğitimi
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

ROOT = Path(__file__).resolve().parent
DATASET_CANDIDATES = [ROOT / "final_verisetim.csv"]
MODEL_PATH = ROOT / "tid_holistic_model.keras"
LABELS_PATH = ROOT / "siniflar.npy"

DATA_PATH = next((path for path in DATASET_CANDIDATES if path.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError("final_verisetim.csv bulunamadı. Lütfen dosyayı proje klasörüne koyun.")

print("1. Holistic Veri Seti Yükleniyor ('final_verisetim.csv')...")
df = pd.read_csv(DATA_PATH)

# Etiketleri (Y) ve Koordinatları (X) ayır
X = df.drop("etiket", axis=1).values
y = df["etiket"].values

print("\n2. Etiketler Sayısallaştırılıyor...")
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
sinif_sayisi = len(encoder.classes_)
np.save(LABELS_PATH, encoder.classes_)

class_weight_values = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded,
)
class_weight_dict = dict(zip(np.unique(y_encoded), class_weight_values))

print("\n3. Veri LSTM Formatına Dönüştürülüyor (Reshaping)...")
# Her karede artık 258 koordinat var (Pose + Eller)
ZAMAN_ADIMI = 20
OZELLIK_SAYISI = 258 
X_reshaped = X.reshape(-1, ZAMAN_ADIMI, OZELLIK_SAYISI)

print(f"-> Yeni Veri Boyutu: {X_reshaped.shape} (Örnek, Zaman, Koordinat)")

print("\n4. Eğitim ve Test Verileri Ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped,
    y_encoded,
    test_size=0.1,
    random_state=42,
    stratify=y_encoded,
)

print("\n5. Bütüncül LSTM Mimarisi Kuruluyor...")
model = Sequential()

# 4 Katmanlı Derin LSTM Mimarisi
model.add(LSTM(256, return_sequences=True, input_shape=(ZAMAN_ADIMI, OZELLIK_SAYISI)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(LSTM(128, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(LSTM(64, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(LSTM(32, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dense(sinif_sayisi, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Eğitim daha derin olduğu için sabrı (patience) biraz artırdık
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

print("\n6. Eğitim Başlıyor (Holistic veri daha ağır olduğu için CPU'yu biraz daha yorabilir)...")
history = model.fit(
    X_train, y_train, 
    epochs=120, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weight_dict,
)

print("\n7. Holistic Model Kaydediliyor...")
model.save(MODEL_PATH)
print("--- MÜTHİŞ! BÜTÜNCÜL MODEL BAŞARIYLA KAYDEDİLDİ: 'tid_holistic_model.keras' ---")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nModelin Holistic Test Başarısı: %{test_acc * 100:.2f}")