# Support Ticket Router (Streamlit)

Small ML app that predicts a support **queue** from ticket text (subject + body).

## What it does
- Input: free-form support ticket text (subject + body).
- Output: predicted queue + confidence + top-K alternatives.
- UI: Streamlit app with example buttons and a simple explanation of influential tokens.

## Dataset
This project uses the public dataset **Tobi-Bueck/customer-support-tickets** (CSV) from Hugging Face:
https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets

Quick download used in this repo:
https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main/dataset-tickets-multi-lang-4-20k.csv

The training script expects these columns in `data/dataset.csv`:
- `subject`
- `body`
- `queue`

## Project structure
```text
support-ticket-router/
  app.py
  train.py
  requirements.txt
  README.md
  data/
    dataset.csv
  artifacts/
    model.joblib


## How to run 

- Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

- Install dependencies
pip install -r requirements.txt

- Download data
mkdir -p data
curl -L -o data/dataset.csv \
  https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main/dataset-tickets-multi-lang-4-20k.csv

- Train model (creates artifacts/model.joblib)
python train.py

- Run UI
python -m streamlit run app.py
