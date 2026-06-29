# Going for Gold: Determining Which Attributes Predict Olympic Medals

**Austin Domer, Hanan Ali, Peyton Hansen**

## Overview
An end-to-end machine learning project analyzing 70,000+ Olympic athlete records to determine which physical and demographic attributes (age, height, weight, sex) predict medal outcomes. The project covers exploratory data analysis, feature selection, class imbalance handling, and comparison of three classification models including a deliberate before/after analysis of how addressing class imbalance changes model performance.

## Research Question
Can an athlete's physical attributes alone predict whether they will win an Olympic medal?

## Dataset
- **Source:** Olympic Athletes and Results — Kaggle
- **Size:** 70,000+ athlete records, filtered to 1980–2016 for modern-era consistency
- **Features used:** Age, Height, Weight, Sex (encoded)
- **Target:** Medal (binary: medaled vs. did not medal)

## Project Structure
```going-for-gold/

├── notebook/

│   └── olympic_medal_prediction.ipynb

├── data/

│   └── dataset_olympics.csv

├── requirements.txt

└── README.md```


## Pipeline

### 1. Data Cleaning
- Removed duplicate records
- Filtered to 1980 and later for modern-era consistency
- Dropped non-predictive identifiers (ID, Name, Games, Year, Season, Sport, NOC, City, Event)
- Explored `Team` visually (top 25 countries by medal count) before dropping it ahead of modeling
- Encoded Sex as a binary numeric feature
- Binary-encoded the Medal column (1 = medaled, 0 = did not medal)

### 2. Exploratory Data Analysis
- Top medal-winning countries
- Distribution of athlete age, height, and weight
- Correlation between physical attributes and medaling
- Gender breakdown of participants and medaling by sex
- Bivariate analysis: scatter plot matrix across numeric variables

### 3. Feature Selection
Chi-squared test ranked features by predictive power:

| Feature | Chi² Score |
|---|---|
| Weight | 1071.79 |
| Height | 277.23 |
| Age | 49.45 |

### 4. Baseline Models — The Misleading Result
| Model | Accuracy |
|---|---|
| Logistic Regression | 86.30% |
| Decision Tree | 82.24% |
| Naive Bayes | 86.30% |

All three baseline models achieved ~86% accuracy by predicting "no medal" for every athlete. The confusion matrices confirmed this: zero medals correctly predicted across all models. This was the central analytical problem motivating the rest of the project.

### 5. Addressing Class Imbalance
Applied `class_weight='balanced'` to Logistic Regression and Decision Tree, and `priors=[0.5, 0.5]` to GaussianNB (switched from MultinomialNB, which doesn't support negative or non-count input). Re-evaluated using precision, recall, and F1-score instead of raw accuracy.

| Model | Accuracy | Medal Recall | Medal Precision | Medal F1 |
|---|---|---|---|---|
| Logistic Regression | 55.97% | 0.72 | 0.20 | 0.32 |
| Decision Tree | 63.25% | 0.62 | 0.22 | 0.33 |
| Naive Bayes | 40.95% | **0.88** | 0.18 | 0.30 |

**Key tradeoff:** correcting for class imbalance significantly lowered overall accuracy but raised medal recall from 0.00 to as high as 0.88. For this problem, recall on the minority class is the more meaningful metric. A model that never predicts a medal is useless even at 86% accuracy.

## Planned Improvements
- [ ] Re-introduce Sport and Country/NOC as encoded features: likely the strongest predictors, set aside in this version to keep the research question focused on physical attributes alone
- [ ] Add Random Forest and XGBoost for comparison
- [ ] Tune classification threshold to balance precision/recall depending on use case
- [ ] Build historical trend visualizations, medal counts by country over decades, sport parity analysis

## Requirements

``` pandas

numpy

scikit-learn

matplotlib

seaborn

imbalanced-learn```

## Install: 
```pip install -r requirements.txt```

## Key Insight
Physical attributes alone are weak predictors of Olympic medals, and the project's most important finding isn't a model, it's a methodological one: ~86% accuracy looked strong but was reporting zero correct medal predictions. Correcting for class imbalance traded accuracy for actual predictive signal on the class that matters, surfacing the real limitation of the feature set rather than masking it. The next step is adding sport and country context which is a deliberate scope decision, not an oversight.
