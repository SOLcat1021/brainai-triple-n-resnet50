# BrainAI: Triple-N ResNet-50 20-ms Mainline

The mainline is a ResNet-50 neural encoding pipeline for Triple-N using official 20-ms response windows. Large datasets, model weights, and generated caches stay outside Git and are supplied through local paths.

## Interpretable framework

The code is organized around the frozen encoder and downstream interpretation audits:

1. `train_all_units_final_resnet50.py`: ResNet-50 + continuous fwRF + depth gate + signed ridge encoding, using official 20-ms neural targets.
2. Downstream temporal, TCAV, and network-dissection scripts are exploratory analyses over the frozen encoder.

The encoder expects the released 20-ms Triple-N population cache and a separately generated 20-ms noise-ceiling table. The legacy 10-ms table is rejected explicitly.

## Data policy

Data, pretrained weights, and generated caches are external local dependencies. The audit manifest records which required artifacts are available.
