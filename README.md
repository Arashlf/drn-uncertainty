# drn-uncertainty

Pipeline for the DRN (distributed reproduction numbers) uncertainty analysis.

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

## Data & citation

- Population: U.S. Census Bureau, [Vintage 2020 County Population Totals](https://www.census.gov/programs-surveys/popest/technical-documentation/research/evaluation-estimates/2020-evaluation-estimates/2010s-counties-total.html)
- Commuting flows: U.S. Census Bureau, [2016–2020 ACS 5-Year Commuting Flows](https://www.census.gov/data/tables/2020/demo/metro-micro/commuting-flows-2020.html)
- DRN concept: She, Paré & Hale, *Distributed reproduction numbers of networked epidemics*, ACC 2023 (doi: [10.23919/ACC55779.2023.10156093](https://doi.org/10.23919/ACC55779.2023.10156093))

BibTeX entries in [`CITATION.bib`](CITATION.bib).
