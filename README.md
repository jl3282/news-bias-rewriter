# News Bias Rewriter & Detector

An advanced system for detecting and rewriting political bias in news articles using fine-tuned transformer models.

## 🎯 Project Overview

This project combines **bias detection** and **content rewriting** to:
- Analyze news articles for political bias (left/center/right)
- Visualize bias on a political spectrum
- Provide bias-adjusted content rewriting
- Serve as an educational tool and research prototype

## 📊 Current Performance

**Fine-tuned RoBERTa Model (59K+ articles):**
- **Overall Accuracy**: 59.7%
- **F1 Score (Macro)**: 56.4%
- **Center Class F1**: 47.1% (significantly improved from baseline)

## 📁 Repository Structure

```
news-bias-rewriter/
├── src/                    # Main source code
│   ├── train_combined_dataset.py    # Combined dataset training (59K articles)
│   ├── roberta_finetuning.py        # Original RoBERTa fine-tuning
│   └── evaluate_trained_model.py    # Model evaluation & testing
├── scripts/                # Utility scripts
│   ├── performance_test.py          # CPU vs MPS performance testing
│   └── input_handler.py             # Text processing utilities
├── models/                 # Trained models
│   └── roberta-bias-classifier/     # Fine-tuned RoBERTa model (59.7% accuracy)
├── data/                   # Datasets
│   ├── jsons/              # JSON dataset (37K articles)
│   └── splits/             # Pre-defined train/val/test splits
├── results/                # Training results & evaluations
├── docs/                   # Documentation & analysis
│   └── COMPREHENSIVE_MODEL_ANALYSIS.md
├── archive/                # Previous experiments
└── README.md
```

## 🚀 Quick Start

### Prerequisites
```bash
# Create conda environment
conda create -n news-bias python=3.9
conda activate news-bias

# Install dependencies
pip install -r requirements.txt
```

### Training Models

#### Option 1: Combined Dataset Training (Recommended)
```bash
# Train on combined AllSides + JSON dataset (59K articles)
cd src/
python train_combined_dataset.py
```

#### Option 2: Original AllSides Training
```bash
# Train on AllSides dataset only (21K articles)
cd src/
python roberta_finetuning.py
```

### Evaluating Models
```bash
# Evaluate trained model
cd src/
python evaluate_trained_model.py
```

### Performance Testing
```bash
# Compare CPU vs MPS/GPU performance
cd scripts/
python performance_test.py
```

## 📊 Datasets

### AllSides Dataset
- **Size**: 21,754 articles
- **Date Range**: November 2022
- **Classes**: Left (47.2%), Right (33.2%), Center (19.6%)
- **Source**: AllSides.com bias ratings

### JSON Dataset  
- **Size**: 37,554 articles
- **Date Range**: 2020 (COVID-19 era)
- **Classes**: Center (33.5%), Left (27.4%), Right (35.7%)
- **Advantage**: Better class balance, especially for center articles

### Combined Dataset (Recommended)
- **Total**: 59,298 articles
- **Balanced Classes**: Center (29.1%), Left (35.6%), Right (35.3%)
- **Key Benefit**: 4x more center examples for improved detection

## 🏗️ Model Architecture

### RoBERTa Fine-tuning
- **Base Model**: FacebookAI/roberta-base (125M parameters)
- **Tokenization**: 512 max tokens, byte-level BPE
- **Training**: 3 epochs, 2e-5 learning rate, batch size 8
- **Optimization**: Research-backed hyperparameters

### Performance Insights
- **CPU Training**: 1.1x faster than MPS for this workload
- **Estimated Training Time**: 11.5 hours for full combined dataset
- **Memory Usage**: ~17GB virtual, 672MB physical during training

## 📈 Key Results & Improvements

### Model Comparison
| Model | Dataset | Accuracy | Center F1 | Training Time |
|-------|---------|----------|-----------|---------------|
| BiasCheck-RoBERTa | AllSides | ~54-60% | ~0% | Pre-trained |
| **Our Fine-tuned** | AllSides | **59.7%** | **47.1%** | 1h 25m |
| **Combined (Target)** | 59K Combined | **70%+** | **65%+** | 11.5h |

### Critical Improvements
1. **Class Balance**: Center representation 19.6% → 29.1%
2. **Scale**: Training data 21K → 59K articles (+180%)
3. **Diversity**: Single time period → Multi-temporal (2020 + 2022)
4. **True 3-Class**: Proper left/center/right vs binary classification

## 🔬 Research Highlights

### Tokenization Optimization
- **512 tokens maximum**: Optimal for news article context
- **BPE subword tokenization**: Handles any Unicode characters
- **Space representation**: Uses Ġ prefix for word boundaries
- **Research-backed**: Based on political bias detection literature

### Training Strategy
- **Fresh model training**: Avoids overfitting to single dataset
- **Stratified splits**: Maintains class balance across train/val/test
- **Conservative settings**: Stable training with batch_size=8
- **Early stopping**: Prevents overfitting with patience=3

## 📝 Usage Examples

### Basic Text Analysis
```python
from scripts.input_handler import load_text_from_file

# Load article
article = load_text_from_file("example.txt")

# Get statistics
stats = get_text_stats(article)
print(f"Words: {stats['word_count']}, Sentences: {stats['sentence_count']}")
```

### Model Evaluation
```python
# Evaluate on custom text
python src/evaluate_trained_model.py --input "Your news article text here"
```

## 🎯 Next Steps

1. **Complete Combined Training**: Train on 59K balanced dataset
2. **Hyperparameter Optimization**: Grid search for optimal settings
3. **Ensemble Methods**: Combine multiple model predictions
4. **Production Deployment**: API endpoint for real-time bias detection
5. **Rewriting Component**: GPT-based bias adjustment system

## 📊 Performance Monitoring

Track model performance using:
- Confusion matrices for class-specific analysis
- F1 scores for balanced performance measurement
- Confidence calibration for prediction reliability
- Cross-dataset evaluation for robustness

## 🤝 Contributing

This project is focused on advancing political bias detection research. Key areas for contribution:
- Dataset expansion and quality improvement
- Model architecture experiments
- Evaluation methodology enhancement
- Production deployment optimization

## 📄 Citation

If you use this work in your research, please cite:
```
News Bias Detection and Rewriting System
Fine-tuned RoBERTa for Political Bias Classification
[Your Name/Institution], 2025
```

---

*For detailed analysis and methodology, see `docs/COMPREHENSIVE_MODEL_ANALYSIS.md`*
