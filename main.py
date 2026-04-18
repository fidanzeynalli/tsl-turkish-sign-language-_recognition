# Faz 7.1: Veri Buyutme ve Koordinat Normalizasyonu (Bilegi 0,0,0 Kabul Etmek)
import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, RepeatVector, TimeDistributed

MAX_FRAME_SAYISI = 128
MAX_PHRASE_LENGTH = 32

def veriyi_hazirla():
    proje_dir = Path(__file__).resolve().parent
    data_dir = proje_dir / "data"

    csv_yolu = data_dir / "train.csv" if (data_dir / "train.csv").exists() else proje_dir / "train.csv"

    print("1. train.csv (Cevap Anahtari) okunuyor...")
    df_csv = pd.read_csv(csv_yolu)

    # --- 1. IYILESTIRME: VERIYI BUYUTME (Coklu Dosya Okuma) ---
    tum_df_eller = []
    # 'data' klasorundeki adi 'file' ile baslayan tum parquet dosyalarini bulur
    parquet_dosyalari = list(data_dir.glob("file*.parquet"))
    
    print(f"\n2. Toplam {len(parquet_dosyalari)} adet veri dosyasi bulundu. Okunuyor...")
    for pq_dosya in parquet_dosyalari:
        print(f" -> {pq_dosya.name} isleniyor...")
        df_pq = pd.read_parquet(pq_dosya, engine="pyarrow")

        # HATA COZUMU: sequence_id eger index (satir etiketi) olarak kayitliysa, onu normal sutuna cevir!
        if 'sequence_id' not in df_pq.columns:
            df_pq = df_pq.reset_index()

        # Kamera ile ayni dili konusmasi icin sutun siralamasini ZORUNLU hale getiriyoruz
        gerekli_sutunlar = ['sequence_id', 'frame']
        for el in ['left_hand', 'right_hand']:
            for eksen in ['x', 'y', 'z']:
                for i in range(21):
                    gerekli_sutunlar.append(f"{eksen}_{el}_{i}")

        df_eller = df_pq[gerekli_sutunlar].fillna(0)
        tum_df_eller.append(df_eller)

    # Tum dosyalari tek bir dev tabloda birlestir
    df_eller_tamami = pd.concat(tum_df_eller, ignore_index=True)
    print(f"--- Veri Birlestirme Tamamlandi (Toplam Satir: {len(df_eller_tamami)}) ---")
    
    return df_csv, df_eller_tamami

def matrisleri_olustur(df_temiz):
    print("\n3. Zaman Standardizasyonu ve NORMALIZASYON baslatiliyor...")
    gruplar = df_temiz.groupby("sequence_id")

    tum_sekanslar = []
    gecerli_idler = []
    koordinat_sutunlari = [col for col in df_temiz.columns if col not in ["frame", "sequence_id"]]

    for seq_id, grup in gruplar:
        ham_matris = grup[koordinat_sutunlari].values
        
        # --- 2. IYILESTIRME: NORMALIZASYON (Bilegi 0'a Sabitleme) ---
        normalize_matris = np.zeros_like(ham_matris)
        
        for i in range(ham_matris.shape[0]):
            kare = ham_matris[i]
            
            # Sol el normalizasyonu (Bilek indeksleri: x=0, y=21, z=42)
            if kare[0] != 0 or kare[21] != 0: 
                sol_x, sol_y, sol_z = kare[0], kare[21], kare[42]
                normalize_matris[i, 0:21] = kare[0:21] - sol_x
                normalize_matris[i, 21:42] = kare[21:42] - sol_y
                normalize_matris[i, 42:63] = kare[42:63] - sol_z
                
            # Sag el normalizasyonu (Bilek indeksleri: x=63, y=84, z=105)
            if kare[63] != 0 or kare[84] != 0:
                sag_x, sag_y, sag_z = kare[63], kare[84], kare[105]
                normalize_matris[i, 63:84] = kare[63:84] - sag_x
                normalize_matris[i, 84:105] = kare[84:105] - sag_y
                normalize_matris[i, 105:126] = kare[105:126] - sag_z

        mevcut_kare = normalize_matris.shape[0]

        if mevcut_kare < MAX_FRAME_SAYISI:
            eksik_kare = MAX_FRAME_SAYISI - mevcut_kare
            sifir_dolgusu = np.zeros((eksik_kare, 126))
            standart_matris = np.vstack((normalize_matris, sifir_dolgusu))
        else:
            standart_matris = normalize_matris[:MAX_FRAME_SAYISI, :]

        tum_sekanslar.append(standart_matris)
        gecerli_idler.append(seq_id)

    X_data = np.array(tum_sekanslar)
    print(f"--- NORMALIZASYON TAMAMLANDI | X Matrisi: {X_data.shape} ---")
    return X_data, gecerli_idler

def etiketleri_olustur(df_csv, gecerli_idler):
    print("\n4. Karakter Kodlamasi (Tokenization) baslatiliyor...")
    df_csv_filtrelenmis = df_csv.set_index("sequence_id").loc[gecerli_idler].reset_index()
    phrases = df_csv_filtrelenmis["phrase"].astype(str).values

    tum_karakterler = list(set("".join(phrases)))
    tum_karakterler.sort()

    char_to_num = {char: i + 1 for i, char in enumerate(tum_karakterler)}

    Y_data = []
    for phrase in phrases:
        sayilar = [char_to_num[char] for char in phrase]
        if len(sayilar) < MAX_PHRASE_LENGTH:
            eksik = MAX_PHRASE_LENGTH - len(sayilar)
            sayilar.extend([0] * eksik)
        else:
            sayilar = sayilar[:MAX_PHRASE_LENGTH]
        Y_data.append(sayilar)

    Y_matrisi = np.array(Y_data)
    print(f"--- TOKENIZATION TAMAMLANDI | Y Matrisi: {Y_matrisi.shape} ---")
    return Y_matrisi, char_to_num

def modeli_kur(X_shape, Y_shape, sozluk_boyutu):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(X_shape[1], X_shape[2])))
    model.add(LSTM(128, return_sequences=False))
    model.add(RepeatVector(Y_shape[1]))
    model.add(LSTM(128, return_sequences=True))
    model.add(TimeDistributed(Dense(sozluk_boyutu + 1, activation='softmax')))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    df_csv, temiz_veri = veriyi_hazirla()
    X_matrisi, gecerli_idler = matrisleri_olustur(temiz_veri)
    Y_matrisi, sozluk = etiketleri_olustur(df_csv, gecerli_idler)
    
    sozluk_boyutu = len(sozluk)
    model = modeli_kur(X_matrisi.shape, Y_matrisi.shape, sozluk_boyutu)

    print("\n5. Model Egitimi (Training) Basliyor...")
    # Veri buyudugu icin Epoch sayisini 30'a cektik
    history = model.fit(X_matrisi, Y_matrisi, epochs=30, batch_size=32, validation_split=0.2)

    model.save("isaret_dili_modeli.keras")
    print("\n--- YENI VE GUCLU MODEL KAYDEDILDI ---")