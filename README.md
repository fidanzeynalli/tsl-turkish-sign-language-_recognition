# Türkçe İşaret Dili (TİD) Gerçek Zamanlı Çeviri Sistemi

Bu proje, MediaPipe tabanlı işaret tespiti ile LSTM zaman serisi sınıflandırmasını birleştiren gerçek zamanlı bir Türkçe İşaret Dili çeviri sistemidir. Canlı kamera akışı `kamera_text.py` üzerinden işlenir, model tahminleri `tf.function` ile hızlandırılır ve sonuçlar çoğunluk oyu ile stabilize edilir.

## Son Mimari

- Canlı tahmin yolu `SEQUENCE_LENGTH = 20` ile çalışır.
- Tahminler `MIN_PREDICTION_FRAMES = 8` sonrası başlatılır.
- Kararlılık için `TAHMIN_TAMPON_BOYUTU = 6` kullanılır.
- Sol ve sağ el bilgisi ayrı hafızalarda tutulur; bir el kaybolduğunda diğer elin verisi korunur.
- Eğitim betiği `normalized_verisetim.csv` dosyasını kullanır ve sınıf dengesizliği için `class_weight` uygular.

## Veri Seti

Bu depoda ağır ham videolar ve büyük eğitim CSV'leri saklanmaz. Eğitim için kullanılan ana dosya `normalized_verisetim.csv` olup, veri toplama tarafında yeni üretim hattı `lstm_veri_topla.py` içindeki `lstm_verisetim_v4.csv` çıktısını oluşturur.

## Çalıştırma

Canlı sistemi başlatmak için:

```bash
python kamera_text.py
```

Eğitim yapmak için:

```bash
python model_egit.py
```