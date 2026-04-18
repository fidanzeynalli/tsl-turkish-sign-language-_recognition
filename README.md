# Türkçe İşaret Dili (TİD) Gerçek Zamanlı Çeviri Sistemi

Bu proje, işitme engelli bireylerle iletişimi kolaylaştırmak amacıyla geliştirilmiş, gerçek zamanlı ve sürekli bir işaret dili tanıma (Continuous Sign Language Recognition) sistemidir. 

Standart statik görüntü sınıflandırmasının ötesine geçerek, **LSTM (Long Short-Term Memory)** yapay sinir ağları ile zaman serisi analizi yapar ve hareketlerin akışını algılar. Model, ezberlemeyi (overfitting) önlemek amacıyla **Burun-Merkezli (Nose-Centric) Koordinat Normalizasyonu** ile eğitilmiş ve kamera geçişlerindeki kararsızlıkları gidermek için **Çoğunluk Oyu (Majority Voting)** stabilite filtresi ile donatılmıştır.

## ⚠️ Veri Seti ve Mimari Notu (Önemli)

Bu projenin yapay zekası, **372 farklı Türkçe İşaret Dili kelimesini** içeren devasa bir veri havuzu üzerinden eğitilmiştir. 

Eğitim sürecinde kullanılan ham videolar (`videolar/` klasörü) ve bu videolardan MediaPipe ile süzülerek çıkarılan Sliding Window koordinat matrisleri (`normalized_verisetim.csv`), toplamda **1.3 GB'ın üzerinde** bir boyuta ulaşmıştır. 

GitHub'ın dosya boyutu sınırlandırmaları (100 MB limiti) ve optimum depo yönetimi (Clean Repository) prensipleri gereğince; **ham videolar ve devasa CSV veri setleri bu GitHub deposuna (.gitignore aracılığıyla) yüklenmemiştir.**

Bu depoda sadece şunlar bulunmaktadır:
1. Sistemin omurgasını oluşturan Python kaynak kodları (`kamera_text.py`, veri toplama ve eğitim algoritmaları).
2. O devasa veri setinden %99.52 başarı oranıyla süzülüp eğitilmiş, kullanıma hazır Yapay Zeka Beyni (`tid_holistic_model.keras` modeli) ve etiket dosyası (`siniflar.npy`).

Projeyi bilgisayarınıza indirip `kamera_text.py` dosyasını çalıştırarak bu gelişmiş mimariyi anında test edebilirsiniz.