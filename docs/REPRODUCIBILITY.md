# Reproducibility checklist

1. Record the exact dataset release, session files, and stimulus archive hashes.
2. Confirm the neural time origin and window labels before fitting.
3. Freeze the pretrained network and response-blind PCA basis.
4. Generate feature caches and verify their documented shapes.
5. Use identical image folds for all compared model variants.
6. Select receptive fields, normalize features, and fit readouts using training images only.
7. Preserve out-of-fold predictions, selected fields, and manifests.
8. Report every tested rank or hyperparameter, not only the winner.
9. For semantic analysis, use native single-layer maps and disjoint discovery/replication image sets.
10. Run the repository audit before using a result in a figure or manuscript.
