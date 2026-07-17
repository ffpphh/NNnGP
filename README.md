# NNnGP experiment submission package

This package collects the code for three experiments and the real-data inputs
for the Montana experiment:

1. tau-sensitivity: weak, median, and strong nonlinearity;
2. neighbor-sine: split and no-split designs;
3. Montana PRISM 800m precipitation with ten train/test split seeds.

## Layout

- `core/`: shared NNnGP model, VI/HMC inference, utilities, and visualization.
- `experiments/01_tau_sensitivity/`: Python experiment code, R NNGP/NNMP
  baselines, and automatic weak/median/strong data generation.
- `experiments/02_neighbor_sine/`: split/no-split simulation generators,
  systematic inference, R baselines, and comparison plotting.
- `experiments/03_montana_800m/`: data preparation, inference, evaluation,
  selected plotting code, R baselines, and the source/split data tables.

All default paths are derived from each script's own location. Python experiment
entry points add the package `core/` directory to their import path, and R
scripts resolve data, helper files, and outputs relative to their script file.
The package can therefore be moved to another directory without editing paths.
Generated results are written under the corresponding experiment's `outputs/`
directory.

The `data/` directories for Experiments 1 and 2 are intentionally not shipped.
Their Python entry points create the directories and generate all required
simulation files automatically. Montana is a real-data experiment, so its input
CSV and split-order table remain included.

Experiment 2 keeps its two Python generators in `python/no_split_simulation/`
and `python/split_simulation/`. Generated data and outputs still use the shorter
scenario names `no_split` and `split`.

## Reproduction entry points

The following examples run from the package root. If a script is invoked by an
absolute path from another working directory, its defaults still resolve from
`__file__` rather than from the current directory.

```bash
# Experiment 1: automatically create data/weak, data/median, data/strong,
# then run VI/HMC according to the configured options
python experiments/01_tau_sensitivity/python/vi_tau_sensitivity.py

# Experiment 2: no-split NNnGP. Missing data are generated automatically.
python experiments/02_neighbor_sine/python/exp_neighbor_sine_systematic_vi.py

# Experiment 2: split NNnGP. The split data directory and files are generated
# automatically when the requested NPZ does not exist.
python experiments/02_neighbor_sine/python/exp_neighbor_sine_systematic_vi.py \
  --dataset split

# Experiment 3: Montana NNnGP, ten seeds
python experiments/03_montana_800m/python/inference/vi_montana_800m_split_seeds.py

# Experiment 3: summarize the NNnGP seed results
python experiments/03_montana_800m/python/evaluation/summarize_montana_800m_split_seeds.py
```

Set `JAX_PLATFORM=cpu` before a Python command when reproducing without a GPU.
The R scripts likewise have package-relative defaults. Run the corresponding
Python workflow before an R baseline: Experiment 1's Python script generates
the three scenario CSVs, and Experiment 2's systematic Python script exports
`ordered_csv/sine_systematic_data.csv`. Sine R baselines default to the
no-split ordered CSV; the split ordered CSV can be passed as their first
command-line argument.

## R dependencies

- `spNNGP` for latent NNGP baselines.
- `nnmp` for Gaussian NNMP baselines.

The Montana R runners require `montana_helpers_parallel.R`, which is included
in the same `r_baselines/` directory.

## Montana plotting selection

Only plotting code corresponding to the current
`prism_montana_october_ppt_2025_800m_split_seeds` workflow is included:

- seed-43 truth/NNnGP/NNGP/NNMP posterior-mean heatmaps;
- tail-event metric plots created during evaluation;
- lower/upper tail seed-median and quantile-band plots.

Unused global boxplot and older single-split heatmap/probability-map scripts are
not included.
