# Validated local results

The current production PCA16 encoding run covers 47,503 units with five-fold image-wise out-of-fold evaluation.

For the audited MF1 unit 18316 (session 36, local unit 179; neural reliability 0.9728), the five-stage PCA16 model reached a best out-of-fold correlation of 0.4432 at 140-159 ms.

A controlled res5 rank scan froze the fold-local PCA16 receptive-field choices and varied only readout dimensionality. Best out-of-fold correlations were:

- PCA16: 0.3968;
- PCA32: 0.4316;
- PCA64: 0.4572;
- PCA128: 0.4223;
- PCA256: 0.3159.

For this unit and layer, PCA64 provided the best held-out prediction. This is a single-unit diagnostic, not evidence that PCA64 is globally optimal.

Semantic conclusions require a separate, single-layer dissection. Prediction accuracy and concept IoU measure different properties.
