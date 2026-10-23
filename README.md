# Multimodal Behavioural and Purchase-Intent Classification from Event Logs

Code for my M.Sc. Eng. thesis (Rzeszów University of Technology, 2026, graduated with honours):
*"Multimodal behavioural and intent classification of e-commerce users using deep neural networks"* (original title in Polish).

The project treats a user session as a sequence of system-log events and classifies it with two original deep learning architectures that use **early multimodal fusion**: discrete event tokens are fused with continuous signals such as time gaps (Δt) and product price at the input level.

- **BiLSTM Fusion**: bidirectional LSTM with attention pooling
- **LOGTransformer Fusion**: Transformer encoder with a `[CLS]` token

Both models are evaluated in two settings:

1. A **controlled simulation**: a Django trading application with request-logging middleware, under synthetic traffic from a Locust simulator. The simulator has 5 behavioural user profiles and injects noise and class overlap.
2. A **real e-commerce dataset**: REES46 *eCommerce behavior data from multi-category store* on Kaggle. This setting predicts purchase intent (cart abandonment vs. purchase).

> Related publication: T. Rak, **P. Greń**, M. Charytanowicz, *Sequential web user classification from system logs using BiLSTM and TCN*, **Scientific Reports**, 2026. [doi:10.1038/s41598-026-69324-6](https://doi.org/10.1038/s41598-026-69324-6)

---

## Key results

**Real data: purchase-intent classification** (REES46, test set of 39,145 sessions)

| Model | Input modalities | Accuracy | Macro F1 | Purchase recall | Purchase precision |
|---|---|---|---|---|---|
| **BiLSTM Fusion** | events + Δt + price | **0.9689** | **0.9087** | 0.9833 | **0.7249** |
| BiLSTM Fusion | events + Δt | 0.9675 | 0.9037 | 0.9878 | 0.7086 |
| BiLSTM Fusion | events only (ablation) | 0.9670 | 0.9024 | 0.9882 | 0.7052 |
| LOGTransformer Fusion | events + Δt + price | 0.9493 | 0.8638 | 0.9837 | 0.6138 |
| LOGTransformer Fusion | events + Δt | 0.9255 | 0.8158 | 0.9905 | 0.5104 |
| LOGTransformer Fusion | events only (ablation) | 0.9342 | 0.8317 | 0.9911 | 0.5416 |

ROC AUC (events + Δt): BiLSTM Fusion **0.9845**, LOGTransformer Fusion 0.9737.

Takeaways from the ablation study:
- The recurrent model wins on real, highly imbalanced data.
- Adding price acts as a **semantic anchor**. It gives the largest gain for the Transformer (+0.048 macro F1), because it stabilises self-attention over noisy, empty time intervals.

**Simulation: 5-class behavioural classification**

| Model | Accuracy | Macro F1 |
|---|---|---|
| BiLSTM Fusion | 0.9118 | 0.9083 |
| LOGTransformer Fusion | 0.9124 | 0.9089 |
| Flat feature baseline (classical ML) | 0.5083 | 0.4892 |

The analytically computed Bayes error bound of the simulated environment corresponds to **~91.14% accuracy**. Both deep models reach this bound, which means they extract all the information available in the logs.

Every run directory contains its exact config, training log, metrics, confusion matrices and final weights. See `Analysis/*/outputs/`.

---

## Repository structure

```
.
├── Analysis/                         # ML part
│   ├── BiLSTM Fusion/                # model, configs, train_net.py, test_net.py, outputs/
│   ├── LOGTransformerFusion/         # model, configs, train_net.py, test_net.py, outputs/
│   ├── baseline_model/               # flat-feature classical ML baseline
│   ├── helpers/                      # data loaders, solver (optimiser/scheduler), logging, plots
│   ├── docs/                         # pipeline and model documentation (PL)
│   ├── preprocess_data.py            # simulation logs (JSONL) -> PyTorch tensors
│   ├── preprocess_real_data.py       # REES46 CSV -> sessions with Δt      (writes data_real/)
│   └── preprocess_real_data2.py      # REES46 CSV -> sessions with Δt + price (writes data_real_v2/)
├── FinanceApp/                       # research environment
│   └── stockProject - main/
│       ├── stockProject/             # Django REST trading app, Celery/Redis, logging middleware, Docker Compose
│       └── loadTestingApp/           # Locust simulator of behavioural user profiles
├── MDocs/                            # progress reports and run guides (PL)
└── UserManual.md                     # full setup and run manual (PL)
```

Outputs are organised as `outputs/simulation`, `outputs/real/v1 - time`, `outputs/real/v2 - time + price` and `outputs/real/v3 - unimodal`. Only the final weights (`final/model_final.pth`) are versioned. Intermediate checkpoints are not.

---

## Quick start

### 1. ML environment

```bash
cd Analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # torch, numpy, pandas, scikit-learn, matplotlib, seaborn, PyYAML, tqdm
```

### 2. Real-data experiments (default mode of the code)

1. Download `2020-Feb.csv` from the Kaggle dataset *eCommerce behavior data from multi-category store* (REES46) into `Analysis/real_data/`.
2. Build the datasets. The scripts read the first `N_ROWS_TO_READ` rows; adjust it at the top of each script.
   ```bash
   python preprocess_real_data.py     # events + Δt          -> data_real/
   python preprocess_real_data2.py    # events + Δt + price  -> data_real_v2/
   ```
3. Train and evaluate. The commands below show BiLSTM; LOGTransformer works the same way from its own folder.
   ```bash
   cd "BiLSTM Fusion"
   python train_net.py --config configs/config_real_price.yaml --num-gpus 1   # events + Δt + price
   python train_net.py --config configs/config_real.yaml       --num-gpus 1   # events + Δt
   python test_net.py  --weights "outputs/real/v2 - time + price/final/model_final.pth"
   ```
   `test_net.py` finds the run's `config.yaml` automatically. It writes the classification report, metrics, ROC AUC and the ROC curve to `results/`.

**Events-only ablation (v3):** use `config_real.yaml` and uncomment the `torch.zeros_like(...)` lines marked `ABLATION STUDY` in `train_net.py` and `test_net.py`.

### 3. Simulation experiments

**Generate logs.** Start the Docker Compose stack (PostgreSQL, Redis, Django API, Celery worker/beat), then run the Locust scenarios:

```bash
cd "FinanceApp/stockProject - main/stockProject"
cp .env.example .env               # then set SECRET_KEY and DB_PASSWORD
docker compose up -d db redis api worker beat
./start_test.sh parameters.txt     # runs Locust scenarios + container monitor, logs go to ./logs
```

Full details are in [`UserManual.md`](UserManual.md) and [`MDocs/quick_run_guide.md`](MDocs/quick_run_guide.md).

**Preprocess.**

```bash
cd Analysis
python preprocess_data.py --log-dir ./logs --output-dir ./data
```

**Switch the code to simulation mode.** By default the code runs on real data. The simulation variants are kept as commented lines:

| File | Change |
|---|---|
| `*/test_net.py` | uncomment the 5-class `CLASS_NAMES` list and comment `["Abandon", "Purchase"]`; comment the binary ROC AUC block |
| `*/train_net.py` | use `all_train_labels = [item["label"] ...]` (simulation) instead of `item["labels"].item()` (real) |
| config | `BiLSTM Fusion/configs/config_fusion.yaml` or `LOGTransformerFusion/configs/config_transformer.yaml` (5 classes, 4 time features) |

The number of continuous features is read from `MODEL.TIME_FEAT_DIM` in the config: 1 = Δt, 2 = Δt + price, 4 = simulation metrics.

---

## Tech stack

Python · PyTorch · scikit-learn · pandas · NumPy · Django REST Framework · PostgreSQL · Celery · Redis · Locust · Docker Compose

## Citation

```bibtex
@article{rak2026sequential,
  title   = {Sequential web user classification from system logs using BiLSTM and TCN},
  author  = {Rak, Tomasz and Gre{\'n}, Piotr and Charytanowicz, Ma{\l}gorzata},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {10.1038/s41598-026-69324-6}
}
```

## Author

**Piotr Greń**. Thesis supervisor: dr inż. Tomasz Rak, Rzeszów University of Technology.
The simulation environment was inspired by an earlier load-testing application by J. Drabek. It was rebuilt and extended for this thesis: class-based API, new endpoints and models, request logging, and a new behavioural Locust simulator.
