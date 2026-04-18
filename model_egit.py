# Faz 15 (Normalized): Ezber Bozan Burun-Merkezli LSTM Model Eğitimi
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

print("1. Normalize Edilmiş Veri Seti Yükleniyor ('normalized_verisetim.csv')...")
df = pd.read_csv("normalized_verisetim.csv")

X = df.drop("etiket", axis=1).values
y = df["etiket"].values

print("\n2. Etiketler Sayısallaştırılıyor...")
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
sinif_sayisi = len(encoder.classes_)
np.save('siniflar.npy', encoder.classes_)

print("\n3. Veri LSTM Formatına Dönüştürülüyor...")
ZAMAN_ADIMI = 20
OZELLIK_SAYISI = 258 
X_reshaped = X.reshape(-1, ZAMAN_ADIMI, OZELLIK_SAYISI)

print(f"-> Yeni Veri Boyutu: {X_reshaped.shape} (Örnek, Zaman, Koordinat)")

print("\n4. Eğitim ve Test Verileri Ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_encoded, test_size=0.1, random_state=42)

print("\n5. Anti-Overfitting Mimari Kuruluyor...")
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(ZAMAN_ADIMI, OZELLIK_SAYISI)))
model.add(BatchNormalization())
model.add(Dropout(0.4)) # Ezberi bozmak için Dropout oranını %40'a çıkardık

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dense(sinif_sayisi, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

print("\n6. Eğitim Başlıyor (14.636 veri olduğu için sabırlı olmalıyız)...")
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=64, # Daha hızlı eğitmek için batch size'ı artırdık
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

print("\n7. Gerçek Zeka Modeli Kaydediliyor...")
model.save("tid_holistic_model.keras")
print("--- MÜTHİŞ! YENİ MODEL BAŞARIYLA KAYDEDİLDİ: 'tid_holistic_model.keras' ---")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nModelin Gerçek (Ezbersiz) Test Başarısı: %{test_acc * 100:.2f}")