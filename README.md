# Comparative Sentiment Analysis: Baseline vs ICL vs Fine-tuning

A comprehensive comparison of three learning paradigms for Twitter sentiment classification using DistilBERT on the TweetEval dataset.

## 📊 Project Overview

This project evaluates three approaches to sentiment analysis:

1. **Baseline (Zero-shot)**: Pretrained DistilBERT without training
2. **In-Context Learning (Few-shot)**: Prompt-based learning with examples
3. **Fine-tuning**: Supervised learning with class-weighted loss

### Key Findings

| Approach | Macro-F1 | Accuracy | Training Time |
|----------|----------|----------|---------------|
| Baseline | 0.2522 | 0.4630 | 0 sec |
| ICL | 0.2147 | 0.4750 | 0 sec |
| **Fine-tuning** | **0.6794** | **0.6830** | **2m 46s** |

**Result**: Fine-tuning with class-weighted loss achieves **169% improvement** over baseline.

## 🎯 Problem Statement

Twitter sentiment analysis faces two key challenges:

1. **Class imbalance**: Unequal distribution of sentiment classes
2. **Learning paradigm selection**: Choosing between zero-shot, few-shot, or fine-tuning

This project systematically compares these approaches to identify the most effective strategy.

## 📁 Dataset

**TweetEval (Sentiment)**: Benchmark dataset for Twitter sentiment analysis

- **Classes**: 
  - 0: Negative
  - 1: Neutral
  - 2: Positive
- **Splits**:
  - Training: 5,000 samples
  - Validation: 1,000 samples
  - Test: 1,000 samples
- **Imbalance**: Training subset shows moderate class imbalance
  - Class weights [2.15, 0.73, 0.85] computed from 5,000 training samples
  - Negative: underrepresented minority class
  - Neutral: majority class (most frequent)
  - Positive: slightly overrepresented

## 🏗️ Architecture

**Model**: DistilBERT-base-uncased
- Efficient transformer model (40% smaller than BERT)
- Pre-trained on English text
- Fine-tuned for 3-class sentiment classification

## 🔬 Methodology

### 1. Baseline (Zero-Shot)

**Approach**: Use pretrained DistilBERT without any training.

**Implementation**:
```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=3
)
```

**Results**:
- Heavily biased toward majority class (Neutral)
- Cannot detect positive sentiment (0% recall)
- Poor macro-F1: 0.2522

### 2. In-Context Learning (Few-Shot)

**Approach**: Provide labeled examples in the prompt.

**Implementation**:
```python
prompt = """
Tweet: I hate this movie. Sentiment: Negative.
Tweet: This is okay. Sentiment: Neutral.
Tweet: I love this phone. Sentiment: Positive.
Tweet: {test_tweet}. Sentiment:
"""
```

**Results**:
- **Worse than baseline** (macro-F1: 0.2147)
- Predicts only neutral class
- **Key Insight**: ICL doesn't work with encoder-only models like DistilBERT

**Why ICL Failed**:
- DistilBERT is designed for classification, not instruction-following
- Encoder-only architecture incompatible with prompt-based learning
- ICL requires decoder models (GPT) or encoder-decoder models (T5)

### 3. Fine-Tuning (Best Approach)

**Approach**: Train with class-weighted loss to handle imbalance.

**Key Innovation - Class-Weighted Loss**:
```python
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=[0, 1, 2],
    y=train_labels
)
# Result: [2.1505, 0.7345, 0.8521]

loss_fn = nn.CrossEntropyLoss(weight=class_weights)
```

**Training Configuration**:
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Optimizer: AdamW with weight decay


### Per-Class Performance

| Class | Baseline F1 | ICL F1 | Fine-tuned F1 | Improvement |
|-------|-------------|--------|---------------|-------------|
| Negative | 0.1308 | 0.0000 | **0.7155** | +447% |
| Neutral | 0.6260 | 0.6441 | **0.6749** | +8% |
| Positive | 0.0000 | 0.0000 | **0.6478** | +∞% |

### Training Progress

```
Epoch 1: Macro-F1 = 0.5962
Epoch 2: Macro-F1 = 0.6422 (+7.7%)
Epoch 3: Macro-F1 = 0.6695 (+4.3%)
```

Steady improvement indicates learning without overfitting.

## 🔑 Key Insights

### 1. Fine-Tuning is Essential for DistilBERT
- Encoder models require parameter updates to adapt
- Zero-shot performance is poor due to random classification head

### 2. ICL Doesn't Work for Encoder Models
- DistilBERT architecture incompatible with prompt-based learning
- Would work better with GPT or T5 models

### 3. Class Weighting Solves Imbalance
- Standard loss → biased toward majority class
- Weighted loss → balanced performance across all classes

### 4. Evaluation Metrics Matter
- **Accuracy**: 46.3% (baseline) looks acceptable
- **Macro-F1**: 25.2% reveals true poor performance
- Always use macro-F1 for imbalanced classification

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/comparative-sentiment-analysis.git
cd comparative-sentiment-analysis

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Run Full Comparison
```bash
python comparative_sentiment_analysis.py
```

### Individual Approaches
```python
# Baseline
from models import evaluate_baseline
results = evaluate_baseline(test_data)

# In-Context Learning
from models import evaluate_icl
results = evaluate_icl(test_data)

# Fine-tuning
from models import finetune_model
model = finetune_model(train_data, val_data)
results = evaluate_finetuned(model, test_data)
```

## 📊 Reproduce Results

```bash
# Run notebook in Google Colab
# Requires: GPU (Tesla T4 or better)
# Runtime: ~3 minutes total (2m 46s training + setup)

jupyter notebook Comparative_Sentiment_Analysis_TweetEval.ipynb
```

## 📚 Dependencies

```
transformers==4.36.0
datasets==2.16.0
torch==2.1.0
scikit-learn==1.3.2
evaluate==0.4.1
numpy==1.24.3
```

## 🎓 Academic Context

This project demonstrates:
- **Transfer Learning**: Using pretrained language models
- **Learning Paradigms**: Zero-shot vs few-shot vs supervised
- **Imbalanced Classification**: Class weighting techniques
- **Model Evaluation**: Appropriate metrics for classification

Suitable for:
- NLP course projects
- Machine Learning assignments
- Research on learning paradigms
- Industry sentiment analysis applications

