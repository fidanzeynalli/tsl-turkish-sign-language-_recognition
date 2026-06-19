import time
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

print('Loading model...')
model = load_model("tid_holistic_model.keras")
print('Loading data sample...')
df = pd.read_csv("benim_verisetim.csv")
X = df.drop("etiket", axis=1).values
ZAMAN_ADIMI=30; OZELLIK_SAYISI=258
X = X.reshape(-1,ZAMAN_ADIMI,OZELLIK_SAYISI)
sample = X[0:1]

# warmup
for _ in range(5):
    model.predict(sample, verbose=0)

# timed runs
runs = 50
t0 = time.time()
for _ in range(runs):
    model.predict(sample, verbose=0)
t1 = time.time()
print("Avg per-predict (ms):", (t1-t0)/runs*1000)
