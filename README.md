# Defense Member Churn MLOps Project

## Overview
End-to-end MLOps pipeline for predicting **defense industry member churn** for a workforce development defense technology non-profit.

### Features
- Data versioning with Git + DVC (S3 remote)
- Modular pipeline scripts:
  - `data_ingest.py` → staged data
  - `data_validation.py` → schema, missing values, distributions
  - `train_and_tune.py` → training + HPO
  - `evaluate.py` → metrics, model save
- Containerized inference (FastAPI) compatible with AWS SageMaker
- CI/CD with GitHub Actions
- Monitoring (Prometheus/CloudWatch), drift detection, alerting
- Governance (Model Card, Incident Response Playbook)
- Canary / blue-green deployment script

### Baseline Performance (Synthetic Data)
See [`artifacts/metrics.json`](artifacts/metrics.json) for full metrics.

| Metric     | Value  |
|------------|--------|
| ROC-AUC    | 0.686 |
| Precision  | 0.544 |
| Recall     | 0.401 |
| F1-score   | 0.461 |
| Accuracy   | 0.666 |

## Folder Layout
```
defense-member-churn-mlops/
├─ data/
│  ├─ raw/                # input data
│  └─ staged/             # output of ingest
├─ src/                   # pipeline scripts
│  ├─ data_ingest.py
│  ├─ data_validation.py
│  ├─ train_and_tune.py
│  └─ evaluate.py
├─ inference/             # FastAPI app for inference
│  └─ serve.py
├─ monitoring/            # drift detection jobs
│  └─ drift_job.py
├─ governance/            # compliance & incident playbooks
│  ├─ model_card.md
│  └─ incident_playbook.md
├─ deploy/                # deploy & traffic shifting scripts
│  ├─ deploy.sh
│  └─ update_traffic.py
├─ diagrams/
│  ├─ architecture.png
│  ├─ repo_tree.png
│  └─ repo_tree.pdf
├─ model/                 # trained model artifacts
├─ artifacts/             # metrics, reports
├─ Dockerfile
├─ dvc.yaml
├─ requirements.txt
├─ .github/workflows/ci_cd.yml
└─ README.md
```

## How to Run
```bash
# Clone repo
git clone <https://github.com/clevesque20/defense-member-churn-mlops.git>
cd defense-member-churn-mlops

# Install dependencies
pip install -r requirements.txt

# Run pipeline locally
dvc repro
# Deploy to SageMaker
./deploy/deploy.sh
```
