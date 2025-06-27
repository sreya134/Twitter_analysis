Sentiment Analysis Pipeline (Tweet Classification)

A complete end-to-end NLP project for detecting racist/sexist sentiment in tweets.
This pipeline walks through data preprocessing, feature engineering, visualization, modeling, and hyperparameter tuning to build an optimized sentiment classification model.

![1538479344425](https://github.com/user-attachments/assets/e4dca5d3-943d-45cf-a51e-8df2e4253fd4)

---

## Objective

Build a robust sentiment analysis model that can classify tweets as either:
- **0** → Non-racist/Sexist
- **1** → Racist/Sexist

The final model uses **XGBoost + Word2Vec embeddings**, optimized with cross-validation to achieve the best F1-score.

---

## 🔍 Key Components

### 1. Library Imports
Essential libraries for:
- Data handling: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`, `WordCloud`
- NLP: `re`, `nltk`, `gensim`
- Modeling: `scikit-learn`, `xgboost`

---

### 2. Data Preparation
- Load training & test datasets
- Add labels and combine datasets for unified processing

### 3. Text Cleaning
- Remove mentions, punctuation, and special characters
- Tokenize, stem (using **PorterStemmer**), and reconstruct clean tweets

---

### 4. Visualization
Explore the dataset using:
- WordClouds (All tweets, Positive, Negative)
- Top hashtags in both sentiment classes

---

### 5. Feature Engineering

| Technique    | Description                                                      |
|--------------|------------------------------------------------------------------|
| BoW          | Basic frequency representation of words                          |
| TF-IDF       | Weighted word frequency based on rarity across documents         |
| Word2Vec     | Dense embeddings capturing semantic relationships                |
| Doc2Vec      | Embeddings representing full tweet/document context              |

Best performance achieved with **Word2Vec** embeddings.

---

### 6. 🔍 Modeling

Tested multiple classification models:

| Model              | Reason for Use                                   |
|--------------------|--------------------------------------------------|
| Logistic Regression| Fast and interpretable baseline                  |
| SVM                | Effective with high-dimensional sparse features  |
| Random Forest      | Robust to overfitting, handles noisy data well   |
| **XGBoost**        | Best overall performance, flexible and powerful  |

---

### 7. XGBoost Tuning

Used **5-fold cross-validation with early stopping** to tune:

- `max_depth`, `min_child_weight`: Control complexity
- `subsample`, `colsample_bytree`: Add randomness to avoid overfitting
- `eta`: Learning rate for boosting
 Final Parameters:
```python
{
  'objective': 'binary:logistic',
  'max_depth': 8,
  'min_child_weight': 6,
  'subsample': 0.9,
  'colsample_bytree': 0.5,
  'eta': 0.1
}

Best Model: XGBoost + Word2Vec
📉 Evaluation Metric: F1 Score
📊 Best Public Score: 0.703


