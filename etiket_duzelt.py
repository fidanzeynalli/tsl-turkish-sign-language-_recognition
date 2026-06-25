from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_CSV = ROOT / "final_verisetim.csv"

SOURCE_DATASET_CANDIDATES = [
    ROOT / "benim_verisetim.csv",
    ROOT / "normalized_verisetim.csv",
    ROOT / "lstm_verisetim.csv",
    ROOT / "lstm_verisetim_v4.csv",
]


def bul_verisetini() -> Path:
    for aday in SOURCE_DATASET_CANDIDATES:
        if aday.exists():
            return aday
    raise FileNotFoundError(
        "Veri seti bulunamadı. benim_verisetim.csv, normalized_verisetim.csv, "
        "lstm_verisetim.csv veya lstm_verisetim_v4.csv dosyalarından biri gerekli."
    )


def temizle(etiket):
    if isinstance(etiket, str) and "_" in etiket:
        return etiket.split("_")[-1]
    return etiket


def main() -> None:
    print("1. Veri seti yükleniyor...")
    veri_yolu = bul_verisetini()
    print(f"-> Kullanılan veri seti: {veri_yolu.name}")

    df = pd.read_csv(veri_yolu)

    if "etiket" not in df.columns:
        raise ValueError(f"Veri setinde 'etiket' sütunu bulunamadı. Bulunan sütunlar: {list(df.columns)}")

    print("2. Etiketler ID bazlı düzenleniyor...")
    df["etiket"] = df["etiket"].apply(temizle)
    df.to_csv(OUTPUT_CSV, index=False)

    print("3. Dönüştürme tamamlandı.")
    print(f"-> Çıktı dosyası: {OUTPUT_CSV.name}")
    print(f"Bitti! Artık etiketlerin sayısal ID'lerden oluşuyor. Final dosyan: '{OUTPUT_CSV.name}'")


if __name__ == "__main__":
    main()
