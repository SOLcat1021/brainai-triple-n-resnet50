# Methods

## Neural targets

Each unit is represented by its image-wise response in consecutive time windows. Splits are performed over images, so every reported prediction for an image comes from a model that did not use that image for field selection, normalization, or readout fitting.

## Visual features

The encoder is an ImageNet-pretrained ResNet-50. Spatial feature maps are extracted from stem, res2, res3, res4, and res5 and aligned to a common grid. Principal components are fitted without neural responses and therefore do not leak the target into the feature basis.

## Feature-weighted receptive field

For every unit, time window, layer, and outer fold, the training images select a Gaussian spatial field from a prespecified bank. The selected field pools the layer feature map into one vector per image.

## Readout

Pooled features are standardized with training-fold statistics. A signed ridge regression, including an unpenalized intercept, predicts the neural response. Out-of-fold predictions from all folds are concatenated, and Pearson correlation is computed against measured responses.

## Interpretation

Prediction and semantic interpretation are separate stages. Multi-layer concatenation is valid for response prediction. Network Dissection is run separately on native layer maps using a top-activation threshold and dataset-level IoU with Broden masks. TCAV and linear concept probes answer different questions and are not interchangeable with IoU.

## Leakage controls

- image-wise outer folds;
- fold-local receptive-field selection;
- fold-local standardization and ridge fitting;
- response-blind PCA;
- disjoint discovery and replication image sets for semantic tests;
- explicit run manifests and immutable cache shapes.
