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
- Overall Accuracy: 78.1%
- F1 Score (Macro): 78.1% 
- Precision (Macro): 78.3%
- Recall (Macro): 78.1%

**Per-Class Performance:**
- Center: F1=78.5%, Precision=78.2%, Recall=78.7%
- Left: F1=78.1%, Precision=75.0%, Recall=81.4%
- Right: F1=77.7%, Precision=81.5%, Recall=74.2%

**Performance Improvement (vs AllSides-only model):**
- Accuracy improved by +18.4 percentage points
- F1 Score improved by +21.7 percentage points  
- Center class F1 improved by +31.4 percentage points

## Training Results

**Latest Training Run (Combined Dataset):**
- Training Duration: 6 hours 15 minutes on CPU
- Training Loss: 0.568 (final)
- Best Validation F1: 78.5% (achieved at epoch 1.93)
- Model convergence: Optimal at 3 epochs (no early stopping triggered)
- Final Test Set Size: 8,895 articles (15% of total dataset)

**Training Configuration:**
- Device: CPU (1.1x faster than MPS for this workload)
- Batch Size: 8
- Learning Rate: 2e-5 with linear decay
- Early Stopping: 3-step patience on validation F1
- Best Model Selection: Based on validation F1 score

**Visualization:** Training analysis plots available in `results/combined_training_analysis.png`

## Repository Structure

```
news-bias-rewriter/
├── app.py                  # Flask web application for bias detection
├── templates/              # HTML templates for web interface
│   └── index.html          # Main web interface template
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

**Dataset Improvements:**
- Training data increased from 21K to 59K articles (180% increase)
- Center class representation improved from 19.6% to 29.1%
- Multi-temporal data coverage (2020 + 2022)
- Achieved balanced 3-class distribution

**Model Performance Gains:**
- Overall accuracy: 59.7% → 78.1% (+18.4%)
- Macro F1 score: 56.4% → 78.1% (+21.7%)
- Center class detection: 47.1% → 78.5% F1 (+31.4%)
- Robust performance across all political bias categories

**Technical Achievements:**
- Optimal training convergence in 3 epochs
- Production-ready model with consistent 78%+ accuracy

## Usage Examples

### Web Interface (Recommended)

**Launch the web application:**
```bash
python app.py
```
Then open your browser to `http://localhost:5001`

**Features:**
- Real-time political bias analysis
- Interactive visualizations showing probability distributions
- Political spectrum positioning
- Example texts for testing
- Responsive design for desktop and mobile

### Command Line Training

**Train the combined dataset model:**
```bash
cd src
python train_combined_dataset.py
```

**Evaluate the trained model:**
```bash
cd src  
python evaluate_trained_model.py
```

**Note:** The latest trained model achieves 78.1% accuracy and is saved to `models/roberta-combined-classifier/` (not pushed due to the size)

## Contributing

Contributions are welcome in the areas of dataset expansion, model architecture, evaluation methodology, and deployment.