import time
import numpy as np
from tensorflow.keras.models import load_model

print('Loading model...')
model = load_model("tid_holistic_model.keras")

# create random sample with expected shape
ZAMAN_ADIMI=30; OZELLIK_SAYISI=258
sample = np.random.rand(1, ZAMAN_ADIMI, OZELLIK_SAYISI).astype(np.float32)

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
