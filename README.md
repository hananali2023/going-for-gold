# Going for Gold: Predicting Olympic Medal Outcomes
**Hanan Ali, Austin Domer, Peyton Hansen**

## Overview

An end-to-end machine learning project analyzing 70,000+ Olympic athlete records to determine which physical and demographic attributes (age, height, and weight) predict medal outcomes. The project covers exploratory data analysis, feature selection, class imbalance handling, and comparison of three classification models.

---

## Research Question

**Can an athlete's physical attributes alone predict whether they will win an Olympic medal?**

---

## Dataset

- **Source:** [Olympic Athletes and Results — Kaggle](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)
- **Size:** 70,000+ athlete records spanning 120 years of Olympic history (1896–2016)
- **Features used:** Age, Height, Weight
- **Target:** Medal (binary: medaled vs. did not medal)

---

## Project Structure

```
going-for-gold/
├── notebook/
│   └── olympic_medal_prediction.ipynb
├── data/
│   └── dataset_olympics.csv
├── requirements.txt
└── README.md
```

---

## Pipeline

### 1. Data Cleaning
- Removed duplicate records
- Dropped non-predictive identifiers (ID, Name, Team, NOC, Event, Sport, City)
- Imputed and dropped rows with missing Age, Height, and Weight values
- Binary-encoded Medal column (1 = medaled, 0 = did not medal)

### 2. Exploratory Data Analysis
- Top medal-winning countries (US leads with 709 gold medals)
- Distribution of athlete age, height, and weight
- Correlation between physical attributes and medaling
- Gender breakdown of participants (M: 36,125 / F: 16,141)

### 3. Feature Selection
Chi-squared test ranked features by predictive power:

| Feature | Chi² Score |
|---------|-----------|
| Weight | 1071.79 |
| Height | 277.23 |
| Age | 49.45 |

### 4. Models Compared

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 86.30% |
| Decision Tree | 82.24% |
| Naïve Bayes | 86.30% |

### 5. Class Imbalance — Key Finding

The dataset is heavily imbalanced: only ~14% of athletes medal. All three baseline models achieve ~86% accuracy by predicting "no medal" for every athlete; a misleading metric. The confusion matrices confirm this: `[[18042, 0], [2865, 0]]` (zero medals correctly predicted).

**This is the central analytical challenge of the project** and motivates the improvements below.

---

## Planned Improvements

- [ ] Apply **SMOTE** or `class_weight='balanced'` to address class imbalance
- [ ] Report **precision, recall, and F1-score** instead of raw accuracy
- [ ] Re-introduce **Sport and Country/NOC** as encoded features - likely the strongest predictors
- [ ] Add **Random Forest** and **XGBoost** for comparison
- [ ] Build **historical trend visualizations** — medal counts by country over decades, sport parity analysis

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn
```

Install:
```bash
pip install -r requirements.txt
```

---

## Key Insight

Physical attributes alone are weak predictors of Olympic medals. Weight is the most informative feature by chi-squared score, but the models' failure to predict any medals reveals that the real signal lies in sport-specific and country-level factors not captured by age, height, and weight alone.
