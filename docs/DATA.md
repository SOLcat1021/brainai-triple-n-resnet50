# Data contract

The repository does not redistribute neural recordings, stimulus images, Broden, or pretrained checkpoints.

Expected external inputs include:

```text
data/
  TripleN/
    V1/
      Raw/H5FILES/
      Processed/
      others/
  Broden/
    index.csv
    label.csv
    c_*.csv
    images/
models/
  vision/
cache/
```

Triple-N session files must retain their released session identifiers. Broden must retain the relative image and mask paths referenced by `index.csv`. Generated feature arrays and predictions belong under `cache/`, which is ignored by Git.

Before a full run, verify image counts, response dimensions, session alignment, and the response-window offset recorded in the run manifest.
