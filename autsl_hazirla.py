# ==============================================================================
# AUTSL VERİ SETİ KÖPRÜ (ENTEGRASYON) ARACI
# AMAÇ: Kaggle'dan indirilen karmaşık isimli (örn: signer0_sample1) videoları 
# okuyup, etiketlerine göre Türkçeleştirerek (örn: anne_1) ana projeye entegre etmek.
# ==============================================================================

import os
import pandas as pd
import shutil
import re

print("--- AUTSL Veri Seti Hazırlayıcı Başlıyor ---\n")

# 1. KAGGLE'DAN İNDİRİLEN VERİLERİN YOLLARI (Terminal çıktına göre güncellendi)
AUTSL_VIDEO_KLASORU = "autsl_data/train"  # Videoların bulunduğu klasör
LABELS_CSV_YOLU = "autsl_data/train_labels.csv" # Hangi videonun hangi numaralı sınıfa ait olduğunu gösteren dosya
CLASS_ID_CSV_ADAYLARI = [
    "autsl_data/SignList_ClassId_TR_EN.csv",
    "autsl_data/class_ids.csv",
    "autsl_data/class_id.csv",
]

CLASS_ID_CSV_YOLU = next((yol for yol in CLASS_ID_CSV_ADAYLARI if os.path.exists(yol)), None)

HEDEF_KLASOR = "videolar" # Projenin yapay zekayı eğiteceği nihai klasör

# Hedef klasör yoksa oluşturuluyor
if not os.path.exists(HEDEF_KLASOR):
    os.makedirs(HEDEF_KLASOR)

# ==============================================================================
# TÜRKÇE KARAKTER TEMİZLEYİCİ
# Etiketlerdeki (boşanmak, ağaç) gibi kelimeleri bilgisayarın hata vermemesi için
# ingilizce karakterlere (bosanmak, agac) dönüştürür.
# ==============================================================================
def turkce_karakter_temizle(metin):
    degisim = str.maketrans("ğĞıİşŞöÖüÜçÇ ", "gGiIsSoOuUcC_")
    metin = str(metin).translate(degisim).lower()
    metin = re.sub(r'[^a-z_]', '', metin)
    return metin

print("1. Etiket Haritaları Okunuyor...")

# Videoları sınıf numaralarına (0, 1, 2...) bağlayan ana haritayı oku
df_labels = pd.read_csv(LABELS_CSV_YOLU, header=None, names=['VideoName', 'ClassId'])

sinif_sozlugu = {}
# Eğer kelime sözlüğü dosyası varsa, numaraları Türkçe kelimelere çevir
if CLASS_ID_CSV_YOLU:
    df_classes = pd.read_csv(CLASS_ID_CSV_YOLU)
    sinif_sozlugu = dict(zip(df_classes['ClassId'], df_classes['TR']))
else:
    # DİKKAT: Kaggle reposunda bu sözlük bazen olmaz, bu durumda programı çökertmemek 
    # için kelimeleri sinif_0, sinif_1 olarak isimlendiririz. (Model yine kusursuz çalışır)
    print("\n[BİLGİ] Sınıf sözlüğü CSV dosyası bulunamadı.")
    print("Videolar 'sinif_0', 'sinif_1' şeklinde numaralandırılarak kopyalanacak.\n")

print("2. Videolar Kopyalanıyor ve Yeniden İsimlendiriliyor...")
sayaclar = {}
kopyalanan_sayisi = 0
bulunamayan_sayisi = 0

# ==============================================================================
# BÜYÜK DÖNGÜ: VİDEOLARI BUL VE YENİDEN İSİMLENDİREREK KOPYALA
# ==============================================================================
for index, row in df_labels.iterrows():
    video_ham_isim = str(row['VideoName']).strip()
    sinif_id = row['ClassId']
    
    # Sınıf numarasını kelimeye çevir (Sözlük yoksa sinif_id kullan)
    turkce_kelime = sinif_sozlugu.get(sinif_id, f"sinif_{sinif_id}")
    temiz_etiket = turkce_karakter_temizle(turkce_kelime)
    
    # Aynı kelimeden kaç tane kopyaladığımızı tut (Örn: anne_1, anne_2...)
    sayaclar[temiz_etiket] = sayaclar.get(temiz_etiket, 0) + 1
    yeni_isim = f"{temiz_etiket}_{sayaclar[temiz_etiket]}.mp4"
    
    # Kaggle dosyalarında ".mp4" veya "_color.mp4" varyasyonları olabilir, ikisini de dene
    kaynak_yol = os.path.join(AUTSL_VIDEO_KLASORU, f"{video_ham_isim}.mp4")
    if not os.path.exists(kaynak_yol):
        kaynak_yol = os.path.join(AUTSL_VIDEO_KLASORU, f"{video_ham_isim}_color.mp4")
        
    hedef_yol = os.path.join(HEDEF_KLASOR, yeni_isim)
    
    # Eğer video bulunduysa, 'videolar' klasörüne yeni ve Türkçe ismiyle kopyala
    if os.path.exists(kaynak_yol):
        shutil.copy2(kaynak_yol, hedef_yol)
        kopyalanan_sayisi += 1
        if kopyalanan_sayisi % 500 == 0:
            print(f" -> {kopyalanan_sayisi} video başarıyla dönüştürüldü...")
    else:
        bulunamayan_sayisi += 1

print(f"\n--- İŞLEM TAMAMLANDI ---")
print(f"Başarıyla Kopyalanan Video: {kopyalanan_sayisi}")
print(f"Bulunamayan/Eksik Video: {bulunamayan_sayisi}")