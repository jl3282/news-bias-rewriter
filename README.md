# News Bias Rewriter & Detector

A system for detecting and rewriting political bias in news articles using fine-tuned transformer models.

## Project Overview

This project combines bias detection and content rewriting to:
- Analyze news articles for political bias (left/center/right)
- Visualize bias on a political spectrum
- Provide bias-adjusted content rewriting
- Serve as an educational tool and research prototype

## Current Performance

**Fine-tuned RoBERTa Model (59K+ articles):**
- Overall Accuracy: 59.7%
- F1 Score (Macro): 56.4%
- Center Class F1: 47.1%

## Repository Structure

```
news-bias-rewriter/
├── src/                    # Main source code
│   ├── train_combined_dataset.py    # Combined dataset training (59K articles)
│   ├── roberta_finetuning.py        # RoBERTa fine-tuning
│   └── evaluate_trained_model.py    # Model evaluation & testing
├── scripts/                # Utility scripts
│   ├── performance_test.py          # CPU vs MPS performance testing
│   └── input_handler.py             # Text processing utilities
├── models/                 # Trained models (not tracked in repo)
├── data/                   # Datasets (CSV and JSON, not tracked in repo)
│   ├── allsides_balanced_news_headlines-texts.csv  # Not pushed due to size
│   ├── jsons/              # JSON dataset (37K articles)
│   └── splits/             # Pre-defined train/val/test splits
├── results/                # Training results & evaluations
├── docs/                   # Documentation & analysis
├── archive/                # Previous experiments
└── README.md
```

**Note:** The AllSides CSV file and other large data files are present locally in the `data/` folder but are not pushed to the repository due to their size. Model files are also not tracked in the repository for the same reason.

## Setup

### Prerequisites
```bash
conda create -n news-bias python=3.10
conda activate news-bias
pip install -r requirements.txt
```

## Datasets

### AllSides Dataset
- Size: 21,754 articles
- Date Range: November 2022
- Classes: Left (47.2%), Right (33.2%), Center (19.6%)
- Source: AllSides.com bias ratings
- Not tracked in repository due to size

### JSON Dataset
- Size: 37,554 articles
- Date Range: 2020 (COVID-19 era)
- Classes: Center (33.5%), Left (27.4%), Right (35.7%)
- Not tracked in repository due to size

### Combined Dataset
- Total: 59,298 articles
- Balanced Classes: Center (29.1%), Left (35.6%), Right (35.3%)

## Model Architecture

- Base Model: FacebookAI/roberta-base (125M parameters)
- Tokenization: 512 max tokens, byte-level BPE
- Training: 3 epochs, 2e-5 learning rate, batch size 8

## Results & Improvements

- Center class representation improved from 19.6% to 29.1%
- Training data increased from 21K to 59K articles
- Multi-temporal data (2020 + 2022)
- True 3-class classification (left/center/right)

## Usage Examples

```python
from scripts.input_handler import load_text_from_file
article = load_text_from_file("example.txt")
# ... further processing ...
```

## Contributing

Contributions are welcome in the areas of dataset expansion, model architecture, evaluation methodology, and deployment.

## Citation

If you use this work in your research, please cite:
```
News Bias Detection and Rewriting System
Fine-tuned RoBERTa for Political Bias Classification
[Your Name/Institution], 2025
```
