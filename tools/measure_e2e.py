import time
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# Settings
NUM_FRAMES = 200
SEQUENCE_LENGTH = 20
MIN_PREDICTION_FRAMES = 10
MODEL_PATH = 'tid_holistic_model.keras'
HAND_MODEL = 'hand_landmarker.task'
POSE_MODEL = 'pose_landmarker.task'

# Load model
print('Loading Keras model...')
model = load_model(MODEL_PATH)


@tf.function(reduce_retracing=True)
def predict_step(input_batch):
    return model(input_batch, training=False)


_warmup_input = tf.zeros((1, SEQUENCE_LENGTH, 258), dtype=tf.float32)
predict_step(_warmup_input)

# Prepare MediaPipe
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=VisionRunningMode.IMAGE, num_hands=2
)
pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=VisionRunningMode.IMAGE, min_pose_detection_confidence=0.3
)
hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)

# Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Failed to open camera')
    exit(1)

# Stats
detect_times = []
preproc_times = []
predict_times = []
total_times = []
predict_calls = 0

sekans_hafizasi = []
son_bilinen_koordinat = np.zeros(258, dtype=np.float32)

print('Starting capture for', NUM_FRAMES, 'frames...')
frame_count = 0
try:
    while frame_count < NUM_FRAMES:
        t_frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # detect
        t0 = time.perf_counter()
        pose_res = pose_detector.detect(mp_image)
        hand_res = hand_detector.detect(mp_image)
        t1 = time.perf_counter()
        detect_times.append((t1 - t0) * 1000)

        # preprocess -> coordinates
        t0 = time.perf_counter()
        pose = np.zeros(33 * 4, dtype=np.float32)
        burun_x = burun_y = burun_z = 0.0
        if pose_res.pose_landmarks:
            lm_list = pose_res.pose_landmarks[0]
            burun_x, burun_y, burun_z = lm_list[0].x, lm_list[0].y, lm_list[0].z
            pose = np.array([[lm.x - burun_x, lm.y - burun_y, lm.z - burun_z, lm.visibility] for lm in lm_list], dtype=np.float32).flatten()

        lh = np.zeros(21 * 3, dtype=np.float32)
        rh = np.zeros(21 * 3, dtype=np.float32)
        el_var_mi = False
        if hand_res.hand_landmarks:
            el_var_mi = True
            for i, landmarks in enumerate(hand_res.hand_landmarks):
                turu = hand_res.handedness[i][0].category_name
                flat = np.array([[lm.x - burun_x, lm.y - burun_y, lm.z - burun_z] for lm in landmarks], dtype=np.float32).flatten()
                if turu == 'Left': lh = flat
                else: rh = flat

        koordinatlar = np.concatenate([pose, lh, rh]).astype(np.float32)
        if el_var_mi:
            son_bilinen_koordinat = koordinatlar
        elif np.sum(son_bilinen_koordinat) != 0:
            koordinatlar = son_bilinen_koordinat
        t1 = time.perf_counter()
        preproc_times.append((t1 - t0) * 1000)

        # append to sequence and optionally predict
        sekans_hafizasi.append(koordinatlar)
        sekans_girdi = sekans_hafizasi[-SEQUENCE_LENGTH:]
        if len(sekans_girdi) < SEQUENCE_LENGTH:
            eksik = SEQUENCE_LENGTH - len(sekans_girdi)
            sekans_girdi = sekans_girdi + [son_bilinen_koordinat for _ in range(eksik)]

        tt0 = time.perf_counter()
        if len(sekans_hafizasi) >= MIN_PREDICTION_FRAMES:
            tahmin_girdisi = tf.convert_to_tensor(np.expand_dims(sekans_girdi, axis=0), dtype=tf.float32)
            _ = predict_step(tahmin_girdisi)
            predict_calls += 1
            tt1 = time.perf_counter()
            predict_times.append((tt1 - tt0) * 1000)
        else:
            tt1 = time.perf_counter()
            predict_times.append(0.0)

        t_frame_end = time.perf_counter()
        total_times.append((t_frame_end - t_frame_start) * 1000)

        frame_count += 1

finally:
    cap.release()
    hand_detector.close()
    pose_detector.close()

# Summarize
print('\n--- E2E TIMINGS SUMMARY ---')
print('Frames measured:', frame_count)
print(f"Avg detect (ms): {np.mean(detect_times):.2f} | median: {np.median(detect_times):.2f}")
print(f"Avg preprocess (ms): {np.mean(preproc_times):.2f} | median: {np.median(preproc_times):.2f}")
# only non-zero predict times
nonzero_preds = [p for p in predict_times if p > 0]
if nonzero_preds:
    print(f"Avg predict (ms): {np.mean(nonzero_preds):.2f} | median: {np.median(nonzero_preds):.2f} | calls: {len(nonzero_preds)}")
else:
    print('No predict calls recorded')
print(f"Avg total per-frame (ms): {np.mean(total_times):.2f} | median: {np.median(total_times):.2f}")
print('Estimated end-to-end FPS:', 1000.0/np.mean(total_times))
