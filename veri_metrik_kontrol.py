import os
import cv2
import numpy as np

# Ham eğitim videolarının bulunduğu ana klasörün yolu
# Kendi bilgisayarındaki klasör ismine göre burayı güncelleyebilirsin (Örn: "videolar" veya "data")
VIDEO_KLASORU = "videolar" 

fps_listesi = []
kare_sayilari_listesi = []

print("📊 Eğitim videoları analiz ediliyor, lütfen bekleyin...")

if os.path.exists(VIDEO_KLASORU):
    # Klasördeki tüm alt dizinleri ve videoları tara
    for root, dirs, files in os.walk(VIDEO_KLASORU):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.M4V')):
                video_yolu = os.path.join(root, file)
                
                # Videoyu OpenCV ile aç
                cap = cv2.VideoCapture(video_yolu)
                
                # Metrikleri oku
                fps = cap.get(cv2.CAP_PROP_FPS)
                toplam_kare = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if fps > 0 and toplam_kare > 0:
                    fps_listesi.append(fps)
                    kare_sayilari_listesi.append(toplam_kare)
                
                cap.release()

    if len(fps_listesi) > 0:
        ortalama_fps = round(np.mean(fps_listesi), 2)
        ortalama_kare = int(np.mean(kare_sayilari_listesi))
        
        print("\n📈 --- EĞİTİM VERİSİ ANALİZ RAPORU ---")
        print(f"🔹 Toplam Taranan Video Sayısı : {len(fps_listesi)}")
        print(f"🔹 Orijinal Eğitim Hızı (FPS)  : {ortalama_fps} FPS")
        print(f"🔹 Ortalama Video Uzunluğu      : {ortalama_kare} Kare")
        print("---------------------------------------\n")
        print("💡 ŞİMDİ NE YAPACAĞIZ?")
        print(f"1. Model eğitimindeki SEQUENCE_LENGTH değerini {ortalama_kare} olarak güncelleyeceğiz.")
        
        # FPS durumuna göre canlı kamera gecikmesini hesapla
        hedef_milisaniye = round((1.0 / ortalama_fps) * 1000, 1)
        print(f"2. Canlı kamerada (kamera_text.py) her kare yakalama arasına TAM {hedef_milisaniye} ms sınır koyacağız.")
    else:
        print("❌ Klasör içinde geçerli bir video dosyası bulunamadı.")
else:
    print(f"❌ '{VIDEO_KLASORU}' isimli klasör bulunamadı. Lütfen ham videolarının olduğu klasör adını yazın.")