# IL-ROM: Case 1

Code for the forced-convection benchmark (Case 1) of the paper *IL-ROM: Identifiability-driven latent reduced-order modeling for cross-field reconstruction under variable observations*. The task is to reconstruct bottom-wall heat flux from sparse temperature and velocity readings under five sensor layouts, C1–C5.

The repository contains IL-ROM and the eight comparison methods of Table 3.4. Cases 2 and 3 of the paper are not included.

## Setup

Use Python 3.10. Install PyTorch first, then the other packages:

```sh
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

For a CPU-only install, use `https://download.pytorch.org/whl/cpu` as the index.

## Data

`T.npy` (655 MB) is too large for this repository, so it is attached to the [v1.0 release](https://github.com/yidongzhang558/IL-ROM/releases/tag/v1.0). The scripts download it to `data/case1_forced_convection/` the first time they need it and check its SHA-256. No manual step is needed.

The other Case 1 files are already in the repository. See `data/case1_forced_convection/README.md` for the format and the splits.

## Run

Fitting the linear estimator takes little time and is a quick way to test the setup:

```sh
python scripts/train_case1_ilrom.py --method ridge --seed 2026 --out experiments/ridge_2026
```

It prints the relative L2 error on C1–C5, which should be 0.0871, 0.1940, 0.1109, 0.0862 and 0.2752.

Train one method:

```sh
python scripts/train_case1_ilrom.py --method ilrom --seed 2026 --out experiments/ilrom_2026
python scripts/train_case1_fixed.py --method fno --seed 2026 --out experiments/fno_2026
python scripts/train_case1_modern.py --method voronoi_fno --seed 2026 --out experiments/voronoi_2026
```

| Script | Methods |
| --- | --- |
| `train_case1_ilrom.py` | `ilrom`, `ridge` |
| `train_case1_fixed.py` | `deeponet`, `pod_gp`, `fno`, `set_encoder` |
| `train_case1_modern.py` | `masked_fno`, `masked_fno_small`, `voronoi_fno`, `vidon` |

Each run writes `metrics.json`, the predictions and the model to its output folder.

Run all nine methods for seeds 2026, 2027 and 2028, 30 jobs in total:

```sh
python scripts/run_reconstruction.py --case 1 --output-root experiments/all --device cuda
```

Add `--dry-run` to print the jobs without starting them. Use `--methods` and `--seeds` to run a subset. VIDON needs CUDA; the other methods also run with `--device cpu`.

## Layout

| Path | Contents |
| --- | --- |
| `src/ilrom_basic/` | Data handling and IL-ROM |
| `src/baselines/` | Comparison methods |
| `configs/` | Training settings |
| `scripts/` | Training entry points and the data check |
| `data/` | Case 1 data, except `T.npy` |

## References for the comparison methods

- DeepONet: Lu et al., Nature Machine Intelligence, 2021. doi:10.1038/s42256-021-00302-5
- FNO: Li et al., ICLR, 2021. arXiv:2010.08895
- POD-GP: Guo and Hesthaven, CMAME, 2018. doi:10.1016/j.cma.2018.07.017
- Voronoi input: Fukami et al., Nature Machine Intelligence, 2021. doi:10.1038/s42256-021-00402-2
- VIDON: Prasthofer et al., arXiv:2205.11404, 2022

Voronoi-FNO and VIDON here are adaptations to this benchmark, not reference implementations of those papers.

## License

The code and the Case 1 data are released under the MIT License. See `LICENSE`.
