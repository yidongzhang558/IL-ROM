# Case 1: forced convection

| File | Shape | Meaning |
| --- | --- | --- |
| `T.npy` | `(1600, 320, 320)` | Temperature in K, axes `(sample, y, x)` |
| `qw.npy` | `(1600, 320)` | Bottom-wall heat flux in W/m² |
| `umean.npy` | `(1600,)` | Mean streamwise speed in m/s |
| `metadata.json` | JSON | Generation settings and summary statistics |
| `layouts.json` | JSON | Coordinates and grid indices of layouts C1–C5 |
| `split.json` | JSON | Sample indices of each split |

`T.npy` (655 MB) exceeds the GitHub file size limit, so it is attached to the [v1.0 release](https://github.com/yidongzhang558/IL-ROM/releases/tag/v1.0). The scripts download it here the first time they need it and check its SHA-256. To check all files afterwards, run `python scripts/verify_data.py`.

All arrays are float32 on a nodal 320-point grid over a 0.1 m square domain. The four velocity readings at normalized heights 0.2, 0.4, 0.6 and 0.8 follow from `6 * umean * y * (1-y)`, so no velocity field file is needed.

Indices 0–799 are used for training, 900–1099 for validation and 1100–1599 for testing. Indices 800–899 are not used.

The mean and standard deviation in `metadata.json` cover the whole dataset and are descriptive only. Models fit their normalization and target POD on the training split.
