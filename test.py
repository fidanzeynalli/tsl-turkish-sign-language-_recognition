# Bu kod, data klasorundeki "file1_.parquet" dosyasini script konumuna gore guvenli bicimde acip
# ilk 50 sutun adini yazdirir.
from pathlib import Path

import pandas as pd

parquet_yolu = Path(__file__).resolve().parent / "data" / "file1_.parquet"

df = pd.read_parquet(parquet_yolu, engine="pyarrow")

# Tüm veri setindeki ilk 50 sütunun ismini ekrana yazdırır
print("Dosyadaki ilk 50 sütun ismi:")
for sutun in df.columns[:50]:
    print(sutun)