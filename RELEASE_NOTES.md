# Release Notes

## Latest Update

- Live inference on the camera path is now wrapped with `tf.function` for lower latency.
- Prediction now starts earlier with `MIN_PREDICTION_FRAMES = 8` and uses a smaller stabilization buffer (`TAHMIN_TAMPON_BOYUTU = 6`).
- Left and right hand fallback memory is tracked separately to reduce occlusion glitches.
- Training now uses `normalized_verisetim.csv` and class balancing with `class_weight`.
- The README was updated to reflect the current 20-frame holistic pipeline and runtime behavior.
