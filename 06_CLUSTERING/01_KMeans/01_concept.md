# K-Means Clustering

## Contents
>- [What is it?](#what-is-it)
>- [How does it work](#how-does-it-work)
>- [Mathematics](#mathematics)
>- [Important Points](#important-points)
>- [When to use it](#when-to-use-it)
>- [Limitations](#limitations)
>- [Key Takeaways](#key-takeaways)
>- [Defer](#defer)



## What is it?
>K-Means is an unsupervised machine learning algorithm used to partition a dataset into \(K\) distinct, non-overlapping clusters. >It groups data points based on similarity, specifically by minimizing the distance between points and the center of the cluster >they belong to. It is widely used for customer segmentation, document clustering, and image compression.
>
## How does it work?
>
### The Core Algorithm (The "Inner Loop")
>K-Means is an iterative algorithm that repeats two main steps until convergence:
>1.  **Choose \(K\):** The user must specify the number of clusters upfront.
>2.  **Initialize Centroids:** Place \(K\) markers in the feature space (randomly or using smart initialization).
>3.  **Assignment Step:** Calculate the distance (usually Euclidean) from every data point to every centroid. Assign each point >to the nearest centroid.
>4.  **Update Step:** Calculate the mean of all points assigned to a cluster and move the centroid to this new mean position.
>5.  **Repeat or Stop:** Repeat Steps 3 and 4 until the centroids stop moving (or hit `max_iter`).
>
### The Problem with Basic K-Means
>Basic K-Means is highly sensitive to the **initial placement of centroids**.
>*   If initial centroids are placed poorly (e.g., two in the same dense blob), the algorithm may converge on a "local optimum" >(a bad clustering).
>*   **Old Solution:** Run the entire algorithm multiple times (e.g., `n_init=10`) with different random starts and keep the >model with the lowest inertia.
>
### K-Means++ (Smart Initialization)
>K-Means++ is not a different algorithm; it is a smarter initialization strategy for Step 2 of the standard algorithm. It solves >the random initialization problem and is now the default in `scikit-learn`.
>1.  **First Centroid:** Pick one data point completely at random from the dataset.
>2.  **Calculate Distances:** Calculate the distance (\(D(x)\)) from every data point to the first centroid.
>3.  **Weighted Probability (The Magic):** Pick the next centroid randomly, but with a probability proportional to \(D(x)^2\). >Points far away from existing centroids have a much higher chance of being chosen.
>4.  **Repeat:** Repeat Step 3 until all \(K\) centroids are chosen.
>5.  **Run Standard K-Means:** Once smart initial centroids are placed, run the standard Assignment and Update loops.
>
## Mathematics
>
>*   **Inertia (Within-Cluster Sum of Squares):** Measures how internally coherent the clusters are. It is the sum of squared >distances between each data point and its assigned centroid.
>    *   **Formula:** $\sum_{i=1}^{n} ||x_i - \mu_{c_i}||^2$
>    *   **Interpretation:** Lower is better. A lower inertia means the clusters are dense and compact.
>*   **Voronoi Tessellation:** The geometric partitioning of a space into regions (Voronoi cells) based on proximity to a set of >points (centroids). Any new point falling into a specific cell is assigned to that cell's centroid.
>## Important Points
>
### Crucial Rules for Data Scaling
>| Data Type | Method | Why? |
>| :--- | :--- | :--- |
>| **Training Data** | `fit_transform()` | Learns the mean/std dev (fit) and applies the scaling (transform). |
>| **Test/New Data** | `transform()` ONLY | Applies the *already learned* mean/std dev to the new data. <br>**NEVER** use >`fit_transform()` on test data (causes Data Leakage). |
>
>*Note on Prediction:* You can pass any random instance to `predict()`, but it **must** be scaled using the exact same scaler >object used during training, and it must have the exact same number of features.
>
### Important Scikit-Learn Defaults & Terminology
>*   **`n_init=10` (The Outer Loop):** The number of times the K-Means algorithm will be run with different centroid seeds. The >final result is the best output of `n_init` consecutive runs in terms of inertia.
>    *   *Note:* In scikit-learn 1.4+, the default changed to `n_init='auto'` (usually 1 if using K-Means++).
>*   **`max_iter=300` (The Inner Loop):** The maximum number of times the Assignment and Update steps will run within a single >execution.
>*   **`init='k-means++'`:** The default initialization method. Spreads initial centroids out to avoid bad local optima.
>*   **`labels_`:** The array of cluster assignments for the training data. Running `predict()` on the training data simply >returns this stored array.
>
### Scikit-Learn Implementation Code
>```python
>from sklearn.cluster import KMeans
>from sklearn.preprocessing import StandardScaler
>import numpy as np
>
># 1. ALWAYS scale your data first!
>scaler = StandardScaler()
>X_train_scaled = scaler.fit_transform(X_train) # Learns AND scales
>
># 2. Train the model
>kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
>kmeans.fit(X_train_scaled)
>
># 3. Inspect the results
>print(kmeans.labels_)          # Array of cluster assignments for training data
>print(kmeans.cluster_centers_) # Array of coordinates for the 5 centroids
>
># 4. Predicting New Data
>X_new = np.array([[50.0, 30.0, 40.0, 0.5], [25.0, 85.0, 75.0, 0.4]])
>X_new_scaled = scaler.transform(X_new) # NOT fit_transform()!
>predictions = kmeans.predict(X_new_scaled)
>print(predictions) # e.g., array([2, 1])
>```
>
See more example at: [02_implementation.ipynb](./02_implementation.ipynb)
## Evaluation Metrics
>
>Evaluating an *unsupervised* algorithm like K-Means is fundamentally different from evaluating a *supervised* algorithm. In >supervised learning, you have "ground truth" labels. In unsupervised learning, you **do not have labels**. Therefore, you cannot >use accuracy, precision, or recall directly.
>
### 1. Internal Metrics (No Ground Truth Needed)
>These are the most common metrics used in real-world unsupervised learning.
>
>*   **Inertia (Within-Cluster Sum of Squares):**
>    *   **Goal:** Lower is better.
>    *   **The Catch (The Elbow Method):** Inertia will *always* decrease as you increase \(K\). You must plot Inertia vs. \(K\) >and look for the "elbow" (the point where the rate of decrease sharply changes from steep to shallow).
>    *   **Scikit-Learn:** `kmeans.inertia_`
>*   **Silhouette Score:**
>    *   **Goal:** Higher is better (Range: \(-1\) to \(+1\)).
>    *   **Interpretation:** Evaluates both **cohesion** (how close a point is to its own cluster) and **separation** (how far a >point is from the nearest neighboring cluster). \(+1\) is excellent, \(0\) is overlapping, \(-1\) is wrong assignment.
>    *   **Scikit-Learn:** `silhouette_score(X_scaled, kmeans.labels_)`
>*   **Davies-Bouldin Index (DBI):**
>    *   **Goal:** Lower is better.
>    *   **Interpretation:** Calculates the average similarity between each cluster and its most similar cluster.
>    *   **Scikit-Learn:** `davies_bouldin_score(X_scaled, kmeans.labels_)`
>*   **Calinski-Harabasz Index:**
>    *   **Goal:** Higher is better.
>    *   **Interpretation:** Ratio of the sum of between-clusters dispersion and within-cluster dispersion.
>    *   **Scikit-Learn:** `calinski_harabasz_score(X_scaled, kmeans.labels_)`
>
### 2. External Metrics (Ground Truth Labels Available)
>Used if you have a dataset with known labels and want to see if K-Means can discover these groupings on its own.
>*   **Adjusted Rand Index (ARI):**
>    *   **Goal:** Higher is better (Range: \(-1\) to \(+1\)).
>    *   **Scikit-Learn:** `adjusted_rand_score(true_labels, kmeans.labels_)`
>*   **Normalized Mutual Information (NMI):**
>    *   **Goal:** Higher is better (Range: \(0\) to \(1\)).
>    *   **Scikit-Learn:** `normalized_mutual_info_score(true_labels, kmeans.labels_)`
>*   **Homogeneity, Completeness, and V-Measure:**
>    *   **Goal:** Higher is better for all three.
>    *   **Scikit-Learn:** `homogeneity_completeness_v_measure(true_labels, kmeans.labels_)`
>
### Summary Table for Evaluation Metrics
>| Metric | Type | Goal | Needs True Labels? | Scikit-Learn Function |
>| :--- | :--- | :--- | :--- | :--- |
>| **Inertia** | Internal | Lower is better | No | `kmeans.inertia_` |
>| **Silhouette** | Internal | Higher is better (Max 1) | No | `silhouette_score()` |
>| **Davies-Bouldin** | Internal | Lower is better | No | `davies_bouldin_score()` |
>| **Calinski-Harabasz** | Internal | Higher is better | No | `calinski_harabasz_score()` |
>| **Adjusted Rand** | External | Higher is better (Max 1) | Yes | `adjusted_rand_score()` |
>| **NMI** | External | Higher is better (Max 1) | Yes | `normalized_mutual_info_score()` |
>| **V-Measure** | External | Higher is better (Max 1) | Yes | `homogeneity_completeness_v_measure()` |
>
## When to use it
>*   When you have unlabeled data and want to discover inherent groupings (e.g., customer segmentation).
>*   When you need a fast, scalable clustering algorithm (it scales linearly with the number of data points).
>*   When you expect your clusters to be roughly spherical (globular) and evenly sized.
>*   When you want a simple, interpretable model (the centroid represents the "average" member of the cluster).
>
## Limitations
>*   **You must choose \(K\) manually:** The algorithm cannot determine the optimal number of clusters on its own.
>*   **Sensitive to initial centroids:** Although K-Means++ fixes most of this, it can still get stuck in local optima on highly >complex datasets.
>*   **Assumes spherical clusters:** K-Means will fail to correctly cluster data with complex shapes (e.g., moons, rings) because >it relies on distance to a center point.
>*   **Sensitive to outliers:** Because it uses the mean, a single extreme outlier can drag a centroid far away from the true >center of the cluster.
>*   **Requires scaling:** Distance-based algorithms are heavily biased by features with larger scales.
>
## Key Takeaways
>*   K-Means partitions data into \(K\) clusters by minimizing the distance between points and their assigned centroids.
>*   **K-Means++** is the modern, default initialization method that dramatically improves results by spreading out initial >centroids.
>*   **Scaling is mandatory** before training. Use `fit_transform()` on training data and `transform()` on new data.
>*   Use the **Elbow Method** (Inertia) to find the optimal \(K\), and validate with the **Silhouette Score** for robust >evaluation.
>*   Never rely on a single metric. Combine mathematical metrics with domain knowledge.
>
## Defer
>- Full mathematical derivation of the Expectation-Maximization (EM) algorithm for Gaussian Mixture Models (a probabilistic >alternative to K-Means).
>- Advanced optimization details for handling massive datasets (e.g., Mini-Batch K-Means).
>- Hierarchical clustering or DBSCAN as alternative clustering paradigms.