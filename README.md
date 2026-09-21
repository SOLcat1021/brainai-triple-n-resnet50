# Time-resolved ResNet-50 encoding of Triple-N neural responses

This repository studies how single-unit representations of natural images evolve over time in the macaque visual pathway. It fits response-blind ResNet-50 features to Triple-N neural responses with image-wise cross-validation, then evaluates semantic structure with concept-based analyses.

## Current scope

- 47,503 neural units across 90 sessions;
- ten consecutive 20-ms response windows;
- ImageNet-pretrained ResNet-50 features from stem through res5;
- Gaussian feature-weighted receptive fields;
- signed ridge readouts with held-out image prediction;
- temporal and semantic audits over frozen encoding models.

Large neural data, model weights, and generated caches are external and are never committed.

## Repository map

- `code/train_all_units_final_resnet50.py`: production neural encoding model.
- `code/select_high_quality_unit_cohorts.py`: reliability and prediction-quality cohorts.
- `code/train_full_10ms_proxy_bank.py`: higher-resolution exploratory proxy bank.
- `code/audit_64d_concept_axis_method.py`: concept-axis audit.
- `code/run_interpretable_framework_audit.py`: artifact and configuration checks.
- `docs/METHODS.md`: model definition and leakage controls.
- `docs/DATA.md`: required external files and directory contract.
- `docs/RESULTS.md`: validated results and interpretation boundaries.
- `docs/REPRODUCIBILITY.md`: end-to-end execution checklist.

## Environment

```bash
conda env create -f environment.yml
conda activate brainai-resnet50
```

CUDA is required for full-scale feature extraction and unit-wise fitting. Small audits can run on CPU.

## Reproduce

1. Place the external Triple-N files according to `docs/DATA.md`.
2. Review paths and frozen constants in `code/interpretable_framework_config.py`.
3. Run the relevant feature extraction and training stages.
4. Run `python code/run_interpretable_framework_audit.py` before interpreting outputs.

The full experiment is computationally intensive. Scripts write manifests and checkpoints so each stage can be audited independently.

## Scientific boundary

The multi-layer model is used for neural-response prediction. Classical Network Dissection must be applied to one native CNN layer at a time; mixing maps from different stages, signed weights, and distinct receptive fields does not produce a standard Network Dissection object.

## Citation

The neural data are from Triple-N. The encoding and interpretation pipeline builds on fwRF encoding, CNN-IF, TCAV, and Network Dissection. See the method document for full references.

## License

Code in this repository is released under the MIT License. Triple-N, Broden, pretrained models, and referenced upstream code retain their original licenses.
