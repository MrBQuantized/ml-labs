# Principal Component Analysis (PCA)

- [What is it?](#what-is-it)
- [How does it work?](#how-does-it-work)
- [Mathematics](#mathematics)
- [Important Points](#important-points)
- [When to use it](#when-to-use-it)
- [Limitations](#limitations)
- [Key Takeaways](#key-takeaways)
- [Defer](#defer)




## What is it?
Principal Component Analysis (PCA) is a widely used **unsupervised** dimensionality reduction technique. Its primary goal is to simplify a dataset by reducing the number of features (dimensions) while preserving as much of the original information (variance) as possible. It does this by finding new, uncorrelated axes (principal components) that align with the directions of maximum variance in the data.

## How does it work?

### The Core Algorithm (Geometric Intuition)
Imagine a 2D scatter plot of data points. PCA tries to find the best line (or plane, or hyperplane) to project these points onto so that the spread of the points is maximized.
1.  **Standardize the Data:** PCA is highly sensitive to scale, so the data must be centered (mean=0) and scaled (variance=1).
2.  **Compute the Covariance Matrix:** Calculate how the features in your dataset vary together.
3.  **Compute Eigenvectors and Eigenvalues:** Mathematically decompose the covariance matrix.
    *   **Eigenvectors:** These represent the *directions* of the new axes (the Principal Components).
    *   **Eigenvalues:** These represent the *magnitude* of variance in each of those directions.
4.  **Sort and Select:** Sort the eigenvectors by their eigenvalues in descending order. Choose the top $(K)$ eigenvectors (where $(K)$ is your desired number of dimensions).
5.  **Project the Data:** Multiply the original data by the top $(K)$ eigenvectors to transform it into the new, lower-dimensional subspace.

### The Problem with Basic PCA
Basic PCA calculates the full covariance matrix, which is computationally expensive ($O(d^2)$ where $(d)$ is the number of features). For datasets with thousands or millions of features, this becomes slow and memory-intensive.
*   **The Solution:** **Randomized PCA** (often the default in `scikit-learn` for large datasets) or **Incremental PCA** (for datasets too large to fit in RAM). Randomized PCA finds approximate eigenvectors much faster.

## Mathematics

*   **Variance Maximization:** PCA seeks the direction $(w)$ that maximizes the variance of the projected data: $\frac{1}{N} \sum_{i=1}^{N} (w^T x_i)^2$. This is equivalent to finding the eigenvector of the covariance matrix with the largest eigenvalue.
*   **Covariance Matrix:** $\Sigma = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)(x_i - \mu)^T$
*   **Eigendecomposition:** $\Sigma = V \Lambda V^T$, where $V$ is the matrix of eigenvectors (principal components) and $\Lambda$ is a diagonal matrix of eigenvalues.
*   **Explained Variance Ratio:** The proportion of the dataset's total variance that is captured by each principal component. It is calculated as $\frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}$.

## Important Points

### Crucial Rules for Data Scaling
| Data Type | Method | Why? |
| :--- | :--- | :--- |
| **Training Data** | `fit_transform()` | Learns the mean/std dev (fit) and applies the scaling (transform). |
| **Test/New Data** | `transform()` ONLY | Applies the *already learned* mean/std dev to the new data. <br>**NEVER** use `fit_transform()` on test data (causes Data Leakage). |

*Note on Prediction:* PCA is not a predictive model; it is a transformation. You can pass any new instance to `transform()`, but it **must** be scaled using the exact same scaler object used during training, and it must have the exact same number of features as the original training data.

### Important Scikit-Learn Defaults & Terminology
*   **`n_components`:** The number of principal components to keep. Can be an integer (e.g., `n_components=2`) or a float between 0 and 1 (e.g., `n_components=0.95` keeps enough components to preserve 95% of the variance).
*   **`svd_solver`:** The algorithm used to compute the PCA. Defaults to `'auto'`, which chooses the best method based on the data size. Options include `'full'`, `'randomized'`, and `'arpack'`.
*   **`explained_variance_ratio_`:** An array showing the percentage of variance explained by each selected component.
*   **`components_`:** The principal axes themselves (the eigenvectors), stored as a matrix.
*   **`inverse_transform()`:** A method to map the reduced data back to the original feature space (though some information is lost).

### Scikit-Learn Implementation Code
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. ALWAYS scale your data first!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Learns AND scales

# 2. Train the PCA model
# Keep 95% of the variance
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)

# 3. Inspect the results
print(pca.explained_variance_ratio_) # Array of variance percentages per component
print(pca.components_)               # The principal axes (eigenvectors)

# 4. Transforming New Data
X_new = np.array([[50.0, 30.0, 40.0, 0.5], [25.0, 85.0, 75.0, 0.4]])
X_new_scaled = scaler.transform(X_new) # NOT fit_transform()!
X_new_pca = pca.transform(X_new_scaled)

# 5. (Optional) Reconstructing the original data
X_reconstructed = pca.inverse_transform(X_new_pca)
```

## Evaluation Metrics

Evaluating PCA is different from evaluating supervised models. Since PCA is a transformation, you evaluate how well it preserves the structure of the original data.

### 1. Internal Metrics (No Ground Truth Needed)
These are the most common metrics used to evaluate PCA.

*   **Explained Variance Ratio (EVR):**
    *   **Goal:** Higher is better for the selected components.
    *   **Interpretation:** This is the primary metric. If you keep 2 components and the EVR is `[0.60, 0.25]`, it means the first component captures 60% of the total variance, and the second captures 25%, for a total of 85%.
    *   **Scikit-Learn:** `pca.explained_variance_ratio_`
*   **Reconstruction Error:**
    *   **Goal:** Lower is better.
    *   **Interpretation:** Measures how much information was lost during dimensionality reduction. It is the mean squared error between the original data and the data reconstructed from the PCA components.
    *   **Scikit-Learn:**
        ```python
        from sklearn.metrics import mean_squared_error
        X_reconstructed = pca.inverse_transform(X_train_pca)
        error = mean_squared_error(X_train_scaled, X_reconstructed)
        print(error)
        ```

### 2. External Metrics (Ground Truth Labels Available)
Used if you have a dataset with known labels and want to see if the PCA components preserve the class separability.
*   **Silhouette Score:**
    *   **Goal:** Higher is better (Range: $-1$ to $+1$).z
    *   **Interpretation:** Measures how well-separated the classes are in the new PCA space.
    *   **Scikit-Learn:** `silhouette_score(X_train_pca, true_labels)`
*   **Adjusted Rand Index (ARI):**
    *   **Goal:** Higher is better (Range: $-1$ to $+1$).
    *   **Scikit-Learn:** `adjusted_rand_score(true_labels, predicted_labels)`

### Summary Table for Evaluation Metrics
| Metric | Type | Goal | Needs True Labels? | Scikit-Learn Function |
| :--- | :--- | :--- | :--- | :--- |
| **Explained Variance** | Internal | Higher is better | No | `pca.explained_variance_ratio_` |
| **Reconstruction Error** | Internal | Lower is better | No | `mean_squared_error()` |
| **Silhouette** | External | Higher is better (Max 1) | Yes | `silhouette_score()` |
| **Adjusted Rand** | External | Higher is better (Max 1) | Yes | `adjusted_rand_score()` |

## When to use it
*   **Visualization:** When you need to plot high-dimensional data in 2D or 3D. (PCA is excellent for this).
*   **Noise Reduction:** By discarding the components with the smallest eigenvalues (which often capture noise), you can improve the performance of downstream models.
*   **Speeding up Training:** Reducing the number of features can dramatically speed up algorithms like Support Vector Machines or K-Nearest Neighbors.
*   **Multicollinearity:** When features are highly correlated, PCA creates new, uncorrelated features (principal components), which is beneficial for linear models like Linear Regression.

## Limitations
*   **Assumes Linearity:** PCA works by finding linear combinations of features. It fails to capture complex, non-linear relationships in the data (e.g., a Swiss roll dataset).
*   **Interpretability Loss:** The new principal components are combinations of the original features, making them much harder to interpret. You can no longer say "Feature 1 is Age" because it's now a mix of Age, Income, and Height.
*   **Sensitive to Scaling:** PCA is extremely sensitive to the scale of the features. If you don't standardize, features with large ranges (e.g., 0-10,000) will dominate the principal components over features with small ranges (e.g., 0-1).
*   **Information Loss:** If you reduce dimensions, you inevitably lose some information. The goal is to lose the least important information, but some is always lost.

## Key Takeaways
*   PCA is a **linear dimensionality reduction** technique that finds the axes of maximum variance.
*   **Scaling is mandatory** before training. Use `fit_transform()` on training data and `transform()` on new data.
*   Use **`explained_variance_ratio_`** to decide how many components to keep (e.g., `n_components=0.95`).
*   PCA is great for visualization, noise reduction, and speeding up training, but it sacrifices interpretability.
*   For non-linear data, consider **Kernel PCA** or **t-SNE** instead.

## Defer
- Full mathematical derivation of Singular Value Decomposition (SVD) and its relationship to PCA.
- Advanced optimization details for handling massive datasets (e.g., Randomized SVD, Incremental PCA).
- Non-linear dimensionality reduction techniques (Kernel PCA, t-SNE, UMAP) as alternative paradigms.