# Repo Blueprint

## Folder Structure

```
01_FUNDAMENTALS/
├── README.md
├── supervised_learning.md
├── unsupervised_learning.md
├── train_val_test.md
├── overfitting_underfitting.md
├── bias_variance.md
└── fundamentals.ipynb

02_MATH_FOR_ML/


03_SCIKIT_LEARN/
├── README.md
├── preprocessing.md
├── pipelines.md
├── model_selection.md
├── metrics.md
└── sklearn_experiments.ipynb

04_REGRESSION/
├── README.md
├── linear_regression.md
├── polynomial_regression.md
├── regularization.md
└── regression_experiments.ipynb

05_CLASSIFICATION/
├── README.md
├── logistic_regression.md
├── knn.md
├── svm.md
├── decision_trees.md
├── confusion_matrix.ipynb
├── classification_threshold.ipynb
└── classification_experiments.ipynb

06_CLUSTERING/
├── README.md
├── kmeans.md
├── dbscan.md
├── hierarchical_clustering.md
├── clustering_metrics.md
└── clustering_experiments.ipynb

07_DIMENSIONALITY_REDUCTION/
├── README.md
├── pca.md
├── tsne.md
└── dimensionality_reduction.ipynb

08_ENSEMBLE_METHODS/
├── README.md
├── bagging.md
├── random_forest.md
├── boosting.md
├── gradient_boosting.md
└── ensemble_experiments.ipynb

09_TIME_SERIES/
├── README.md
├── time_series_fundamentals.md
├── feature_engineering.md
├── forecasting.md
└── time_series_experiments.ipynb

10_MODEL_INTERPRETABILITY/
├── README.md
├── feature_importance.md
├── permutation_importance.md
├── shap.md
├── lime.md
└── interpretability_experiments.ipynb

11_IMBALANCED_LEARNING/
├── README.md
├── class_imbalance.md
├── resampling.md
├── class_weights.md
├── smote.md
└── imbalance_experiments.ipynb

12_RECOMMENDATION_SYSTEMS/
├── README.md
├── collaborative_filtering.md
├── content_based.md
├── similarity.md
└── recommendation_experiments.ipynb

13_DEEP_LEARNING/
├── README.md
├── tensors.md
├── autograd.md
├── neural_networks.md
├── backpropagation.md
├── loss_functions.md
├── optimizers.md
├── training_loops.md
├── cnns.md
├── transfer_learning.md
├── embeddings.md
├── transformers.md
└── notebooks/
    ├── tensors.ipynb
    ├── neural_network.ipynb
    ├── cnn.ipynb
    └── ...

15_PROJECTS/
├── 01_hotel_booking_analysis/
├── 02_url_management/
├── 03_clustering_project/
├── 04_recommendation_system/
└── ...
```
## Project Structure
```
03_clustering_project/
├── README.md
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── clustering.py
│   └── evaluate.py
├── data/
└── reports/
```
---

## Folder READEME.md Blueprints

### 1. `.md` — Concept/Theory Notes

Use this for algorithms, mathematical concepts, ML concepts, and important techniques.

```
# Topic Name

## What is it?

Brief explanation.

## How does it work?

Core intuition / process.

## Mathematics

Key equations and what they mean.

## Important Points

- ...
- ...
- ...

## When to use it

...

## Limitations

...

## Key Takeaways

## Defer
- Full mathematicla derivation
- Advanced optimization details

- ...
- ...
```

### 2. `.ipynb` — Experiment/ Practice
Use this whenever you're actually working with data, implementing something, visualizing results, or comparing models.

```
# Experiment: Topic Name

## Objective

What am I trying to investigate?

## Setup

Imports and configuration.

## Data

Load / inspect / prepare the data.

## Experiment

Implement and test the concept.

## Results

Visualizations, metrics, outputs.

## Interpretation

What happened and why?

## Key Takeaways

- ...
- ...
- ...
```