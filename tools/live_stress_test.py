from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "tid_holistic_model.keras"
LABELS_PATH = ROOT / "siniflar.npy"
HAND_MODEL = ROOT / "hand_landmarker.task"
POSE_MODEL = ROOT / "pose_landmarker.task"

SEQUENCE_LENGTH = 20
MIN_PREDICTION_FRAMES = 8
TAHMIN_TAMPON_BOYUTU = 6
PREDICTION_THRESHOLD = 0.80
TRIALS_PER_WORD = 10
TARGET_WORD_COUNT = 10
WINDOW_NAME = "TID Stres Testi"

print("1. Model ve sınıf listesi yükleniyor...")
model = load_model(MODEL_PATH)
siniflar = np.load(LABELS_PATH, allow_pickle=True).astype(str)
id_sozlugu = {idx: sinif for idx, sinif in enumerate(siniflar)}

if len(siniflar) == 0:
    raise ValueError("siniflar.npy boş görünüyor.")

print(f"-> Toplam sınıf sayısı: {len(siniflar)}")
print("-> İlk 20 sınıf:")
print(list(siniflar[:20]))

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
)
pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=0.3,
)
hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)


@tf.function(reduce_retracing=True)
def predict_step(input_batch):
    return model(input_batch, training=False)


_warmup_input = tf.zeros((1, SEQUENCE_LENGTH, 258), dtype=tf.float32)
predict_step(_warmup_input)


@dataclass
class TrialResult:
    target: str
    predicted: str | None = None
    success: bool = False
    latency_ms: float | None = None


@dataclass
class WordStats:
    target: str
    results: list[TrialResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for result in self.results if result.success)

    @property
    def trial_count(self) -> int:
        return len(self.results)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return 100.0 * self.success_count / self.trial_count

    @property
    def mean_latency_ms(self) -> float:
        latencies = [result.latency_ms for result in self.results if result.latency_ms is not None]
        if not latencies:
            return 0.0
        return float(np.mean(latencies))


BAGLANTILAR = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
]


def koordinatlari_cikar_ve_normalize_et(mp_image, son_sol_el: np.ndarray, son_sag_el: np.ndarray):
    pose_res = pose_detector.detect(mp_image)
    hand_res = hand_detector.detect(mp_image)

    pose = np.zeros(33 * 4, dtype=np.float32)
    burun_x = burun_y = burun_z = 0.0

    if pose_res.pose_landmarks:
        lm_list = pose_res.pose_landmarks[0]
        burun_x, burun_y, burun_z = lm_list[0].x, lm_list[0].y, lm_list[0].z
        pose = np.array(
            [[lm.x - burun_x, lm.y - burun_y, lm.z - burun_z, lm.visibility] for lm in lm_list],
            dtype=np.float32,
        ).flatten()

    lh, rh = son_sol_el.copy(), son_sag_el.copy()
    sol_el_var_mi = False
    sag_el_var_mi = False

    if hand_res.hand_landmarks:
        for i, landmarks in enumerate(hand_res.hand_landmarks):
            turu = hand_res.handedness[i][0].category_name
            flat = np.array(
                [[lm.x - burun_x, lm.y - burun_y, lm.z - burun_z] for lm in landmarks],
                dtype=np.float32,
            ).flatten()
            if turu == "Left":
                lh = flat
                sol_el_var_mi = True
            else:
                rh = flat
                sag_el_var_mi = True

    if sol_el_var_mi:
        son_sol_el = lh
    if sag_el_var_mi:
        son_sag_el = rh

    el_var_mi = sol_el_var_mi or sag_el_var_mi
    return np.concatenate([pose, lh, rh]).astype(np.float32), pose_res, hand_res, el_var_mi, son_sol_el, son_sag_el


def choose_targets() -> list[str]:
    print("\nMevcut sınıflar içinden 10 hedef kelime seç.")
    print("İpucu: İstersen listeyi kopyalayıp aynen yapıştırabilirsin.")
    print("İlk 40 sınıf:")
    print(", ".join(siniflar[:40]))
    raw = input("\n10 kelimeyi virgülle ayırarak gir: ").strip()
    targets = [item.strip() for item in raw.split(",") if item.strip()]
    if len(targets) != TARGET_WORD_COUNT:
        raise ValueError(f"Tam olarak {TARGET_WORD_COUNT} kelime girmen gerekiyor.")
    unknown = [target for target in targets if target not in set(siniflar)]
    if unknown:
        raise ValueError(f"Şu kelimeler model sınıf listesinde yok: {unknown}")
    return targets


def run_trial(target: str) -> TrialResult:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera açılamadı.")

    sekans_hafizasi: list[np.ndarray] = []
    tahmin_tamponu: list[str] = []
    son_bilinen_koordinat = np.zeros(258, dtype=np.float32)
    son_sol_el = np.zeros(21 * 3, dtype=np.float32)
    son_sag_el = np.zeros(21 * 3, dtype=np.float32)

    trial_started = False
    start_time = 0.0
    stable_prediction_time: float | None = None
    predicted_label: str | None = None
    result = TrialResult(target=target)

    print(f"\nHedef: {target}")
    print("İşareti yapmaya başla. Stabil tahmin oluşunca otomatik kaydedilecek.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            dikey_genislik = int(h * (9 / 16))
            baslangic_x = (w // 2) - (dikey_genislik // 2)
            bitis_x = baslangic_x + dikey_genislik
            frame = frame[:, baslangic_x:bitis_x]
            h_new, w_new, _ = frame.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            koordinatlar, pose_res, hand_res, el_var_mi, son_sol_el, son_sag_el = koordinatlari_cikar_ve_normalize_et(
                mp_image,
                son_sol_el,
                son_sag_el,
            )

            sekans_hafizasi.append(koordinatlar)
            sekans_hafizasi = sekans_hafizasi[-SEQUENCE_LENGTH:]

            if len(sekans_hafizasi) >= MIN_PREDICTION_FRAMES:
                sekans_girdi = sekans_hafizasi[-SEQUENCE_LENGTH:]
                if len(sekans_girdi) < SEQUENCE_LENGTH:
                    eksik_kare = SEQUENCE_LENGTH - len(sekans_girdi)
                    sekans_girdi = sekans_girdi + [son_bilinen_koordinat for _ in range(eksik_kare)]

                tahmin_girdisi = tf.convert_to_tensor(np.expand_dims(sekans_girdi, axis=0), dtype=tf.float32)
                res = predict_step(tahmin_girdisi)[0].numpy()
                indeks = int(np.argmax(res))
                guven = float(res[indeks])
                tahmin_kelime = id_sozlugu.get(indeks, "BOS")

                if guven > PREDICTION_THRESHOLD and el_var_mi:
                    tahmin_tamponu.append(tahmin_kelime)
                else:
                    tahmin_tamponu.append("BOS")

                tahmin_tamponu = tahmin_tamponu[-TAHMIN_TAMPON_BOYUTU:]
                if len(tahmin_tamponu) == TAHMIN_TAMPON_BOYUTU:
                    en_cok_tekrar_eden = max(set(tahmin_tamponu), key=tahmin_tamponu.count)
                    tekrar_sayisi = tahmin_tamponu.count(en_cok_tekrar_eden)
                    if en_cok_tekrar_eden != "BOS" and tekrar_sayisi > (TAHMIN_TAMPON_BOYUTU // 2):
                        predicted_label = en_cok_tekrar_eden
                        if not trial_started and predicted_label == target:
                            trial_started = True
                            start_time = time.perf_counter()
                        if trial_started and predicted_label == target and stable_prediction_time is None:
                            stable_prediction_time = (time.perf_counter() - start_time) * 1000.0
                            result.predicted = predicted_label
                            result.success = True
                            result.latency_ms = stable_prediction_time
                            break

            # UI
            cv2.rectangle(frame, (0, 0), (w_new, 50), (245, 117, 16), -1)
            cv2.putText(frame, f"Hedef: {target}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            cv2.rectangle(frame, (0, h_new - 55), (w_new, h_new), (0, 0, 0), -1)
            anlik = predicted_label if predicted_label else "Izleniyor..."
            cv2.putText(frame, f"Tahmin: {anlik}", (10, h_new - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                trial_started = False
                start_time = 0.0
                stable_prediction_time = None
                predicted_label = None
                result.predicted = None
                result.success = False
                result.latency_ms = None
                tahmin_tamponu.clear()
                sekans_hafizasi.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return result


def main() -> None:
    targets = choose_targets()
    word_stats: dict[str, WordStats] = {target: WordStats(target=target) for target in targets}

    print("\nStres test başlıyor.")
    print(f"Her kelime için {TRIALS_PER_WORD} deneme yapılacak.")
    print("İşaret yaparken kameraya bak; tahmin doğru ve stabil olunca deneme otomatik bitecek.")
    print("Bir denemeyi sıfırlamak için 'r', çıkmak için 'q'.")

    for target in targets:
        for trial_index in range(TRIALS_PER_WORD):
            print(f"\n{target} | deneme {trial_index + 1}/{TRIALS_PER_WORD}")
            print("Hazır olduğunda Enter'a bas.")
            input()
            result = run_trial(target)
            word_stats[target].results.append(result)
            status = "BAŞARILI" if result.success else "BAŞARISIZ"
            latency_text = f"{result.latency_ms:.0f} ms" if result.latency_ms is not None else "N/A"
            print(f"Sonuç: {status} | Tahmin: {result.predicted or 'Yok'} | Latency: {latency_text}")

    print("\n--- STRES TESTİ SONUÇLARI ---")
    total_trials = 0
    total_success = 0
    all_latencies = []
    for stats in word_stats.values():
        total_trials += stats.trial_count
        total_success += stats.success_count
        all_latencies.extend([result.latency_ms for result in stats.results if result.latency_ms is not None])
        print(
            f"{stats.target}: başarı {stats.success_count}/{stats.trial_count} "
            f"(%{stats.accuracy:.2f}), ort. gecikme {stats.mean_latency_ms:.0f} ms"
        )

    overall_accuracy = 100.0 * total_success / total_trials if total_trials else 0.0
    overall_latency = float(np.mean(all_latencies)) if all_latencies else 0.0
    print(f"\nGenel başarı: {total_success}/{total_trials} -> %{overall_accuracy:.2f}")
    print(f"Genel ortalama gecikme: {overall_latency:.0f} ms")


if __name__ == "__main__":
    main()
