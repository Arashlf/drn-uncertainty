# drn-uncertainty

Pipeline for the DRN (dynamic reproduction number) uncertainty analysis.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python 1_load_data.py
python 2_run_experiments.py
python 3_make_figures.py
```

Run in that order — each stage reads outputs the previous one wrote to `results/`. Figures land in `results/figures/`.
