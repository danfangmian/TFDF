TFDF

TFDF is a PyTorch Lightning project for multivariate time series forecasting. It trains and evaluates a forecasting model named TFDF, which combines wavelet-based decomposition (via 1D DWT) with a trend modeling branch, and is configured through YAML files using OmegaConf.

Core components

The entrypoint is TFDF/main.py. It loads a base config, merges dataset-specific settings from configs/dataset_conf.yaml, then merges model-specific settings from configs/models/<model>.yaml.
Training and evaluation are implemented with PyTorch Lightning (Trainer, EarlyStopping, ModelCheckpoint).
Data loading is handled by data_provider/datamodules.py, supporting common forecasting datasets such as ETT (hour/minute), and multiple “custom” datasets listed in configs/dataset_conf.yaml.
The TFDF model is implemented in models/forecast/TFDF.py and uses wavelet transforms from the included pytorch_wavelets package (DWT1DForward / DWT1DInverse).

Requirements

Install dependencies from requirements.txt. Main dependencies include torch, pytorch-lightning, omegaconf, numpy, pandas, matplotlib, and optuna.

Setup

1) Create a Python environment (recommended).
2) Install requirements:
   pip install -r requirements.txt

Data preparation

Datasets are configured in configs/dataset_conf.yaml. By default, the config uses ETTh1 with:
root_path: ./data/ETT/
data_path: ETTh1.csv

Place your dataset files under the corresponding root_path and ensure the CSV/TXT file name matches data_path.

Configuration

Main config: configs/conf.yaml
Dataset config map: configs/dataset_conf.yaml
Model config: configs/models/TFDF.yaml

Key forecasting settings are in configs/conf.yaml:
seq_len defines the input history length.
pred_len defines the forecast horizon.
label_len defines the decoder prefix length (used to build decoder inputs).
features controls S / M / MS modes and automatically adjusts input/output dimensions.

Key TFDF model settings are in configs/models/TFDF.yaml:
wavelet and level control wavelet type and decomposition level.
patch_len and stride control patching for the mixer core.
d_model, d_core, n_layers, n_heads control model capacity.
no_decomposition can disable the decomposition pathway.

Training and testing

Run training (and then test with the best checkpoint) using:
python TFDF/main.py --config_path configs/conf.yaml

By default, the script will:
train for up to max_epochs (see configs/conf.yaml)
use early stopping (if enabled)
test using ckpt_path="best"
optionally save test artifacts when save_test_result is true

Test outputs and visualization

When save_test_result is true, the test phase can save:
result.npz containing preds, targets, and input history arrays
PDF plots under a visual/ directory

The visualization sampling behavior is controlled by:
visual_stategy (step, random, lowest, other)
visual_step, visual_num, visual_idxs

Hyperparameter optimization (Optuna)

TFDF/optim.py runs Optuna tuning driven by configs/optim.yaml:
python TFDF/optim.py --config_path configs/conf.yaml --optim_config_path configs/optim.yaml --n_trials 20 --direction minimize --n_jobs 1

The tuning script reports the best trial value and parameters after completion.

Project structure

TFDF/
  main.py                     training and evaluation entrypoint
  optim.py                    Optuna-based hyperparameter tuning
  requirements.txt            Python dependencies
  configs/                    YAML configs for training, dataset, optimizer, and model
  data_provider/              datasets and LightningDataModule for forecasting
  models/                     model definitions and Lightning solver
  pytorch_wavelets/           wavelet transform utilities used by the TFDF model
  utils/                      metrics and helper utilities

Notes

GPU usage is controlled by the trainer params in configs/conf.yaml (accelerator, devices). Adjust these to match your environment.
If you change the dataset name in configs/conf.yaml (data: ...), make sure it exists in configs/dataset_conf.yaml and that the files are present under the expected root_path.
