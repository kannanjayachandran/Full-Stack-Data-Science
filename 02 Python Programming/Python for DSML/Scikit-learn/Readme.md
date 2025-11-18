<!-- 
    Author : Kannan Jayachandran
    File : Readme.md (Scikit-learn (sklearn))
    Section : Python for DSML
 -->

<h1 align="center"> Scikit-learn (sklearn) </h1>

## Introduction

Scikit-learn (sklearn) is Python's most widely adopted machine learning library, providing a unified interface for **supervised** and **unsupervised** learning algorithms, **preprocessing** utilities, **model evaluation tools**, and **pipeline construction**. 

Built on `NumPy`, `SciPy`, and `matplotlib`, it emphasizes ease of use, performance, and accessibility, making it the de facto standard for classical machine learning in production environments. 

Its consistent API design—where estimators implement `fit()`, `predict()`, and `transform()` methods—enables rapid prototyping, model comparison, and deployment of ML solutions across regression, classification, clustering, and dimensionality reduction tasks.

---

## 1. Classification

Classification is a supervised learning task that assigns discrete labels to input samples based on learned patterns from labeled training data. Scikit-learn provides various algorithms including **linear models** (*Logistic Regression*, *SVM*), **tree-based methods** (*Decision Trees*, *Random Forests*, *Gradient Boosting*), **probabilistic models** (*Naive Bayes*), and **instance-based learners** (*KNN*). The library handles binary, multi-class, and multi-label classification scenarios with consistent interfaces.

### 1.1 Logistic Regression

Logistic Regression models the probability of class membership using the **logistic (sigmoid) function**, making it interpretable and efficient for linearly separable data. Despite its name, it's a **classification algorithm** that estimates `P(y=1|X)` through **maximum likelihood estimation**.

**Mathematical Foundation:**
- Sigmoid function: $\sigma(z)=\frac{1}{1 + e^{(-z)}}$
- Decision boundary: $w^T \cdot x + b = 0$
- Loss function: Binary cross-entropy

![alt text](image.png)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize with L2 regularization (default)
log_reg = LogisticRegression(
    penalty='l2',           # Regularization type: 'l1', 'l2', 'elasticnet', 'none'
    C=1.0,                  # Inverse of regularization strength (smaller = stronger)
    solver='lbfgs',         # Optimization algorithm
    max_iter=1000,          # Maximum iterations for convergence
    class_weight='balanced', # Handle imbalanced datasets
    random_state=42
)

# Train model
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)  # Probability estimates

# Model interpretation
print(f"Coefficients shape: {log_reg.coef_.shape}")
print(f"Intercept: {log_reg.intercept_}")
print(f"Classes: {log_reg.classes_}")

# Evaluation
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

**Multi-class Classification Strategies:**

```python
# One-vs-Rest (OvR) - default for most solvers
log_reg_ovr = LogisticRegression(multi_class='ovr')

# Multinomial - single model for all classes (requires multinomial-compatible solver)
log_reg_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Example with multiclass data
from sklearn.datasets import load_iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

log_reg_multi.fit(X_train, y_train)
print(f"Accuracy: {log_reg_multi.score(X_test, y_test):.3f}")
```

**Regularization Comparison:**

```python
from sklearn.preprocessing import StandardScaler

# Feature scaling is crucial for regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# L1 regularization (Lasso) - feature selection via sparsity
log_reg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
log_reg_l1.fit(X_train_scaled, y_train)
print(f"L1 - Non-zero coefficients: {np.sum(log_reg_l1.coef_ != 0)}")

# L2 regularization (Ridge) - coefficient shrinkage
log_reg_l2 = LogisticRegression(penalty='l2', C=0.1)
log_reg_l2.fit(X_train_scaled, y_train)
print(f"L2 - Coefficient norm: {np.linalg.norm(log_reg_l2.coef_):.3f}")

# ElasticNet - combination of L1 and L2
log_reg_en = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.1)
log_reg_en.fit(X_train_scaled, y_train)
```

### 1.2 Support Vector Machines (SVM)

SVMs find optimal hyperplanes that maximize the margin between classes, using kernel tricks to handle non-linear decision boundaries. They're particularly effective in high-dimensional spaces and memory-efficient through support vector representation.

**Key Concepts:**
- **Margin maximization:** Distance between hyperplane and nearest samples
- **Support vectors:** Training samples that define the decision boundary
- **Kernel trick:** Implicit mapping to higher dimensions
- **Soft margin:** Allows misclassifications via slack variables (controlled by C)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate non-linear dataset
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling is CRITICAL for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)
print(f"Linear SVM Accuracy: {svm_linear.score(X_test_scaled, y_test):.3f}")
print(f"Support Vectors: {svm_linear.n_support_}")

# RBF (Radial Basis Function) kernel - most common for non-linear data
svm_rbf = SVC(
    kernel='rbf',
    C=1.0,              # Regularization parameter (smaller = more regularization)
    gamma='scale',      # Kernel coefficient ('scale' = 1/(n_features * X.var()))
    probability=True,   # Enable probability estimates (slower training)
    random_state=42
)
svm_rbf.fit(X_train_scaled, y_train)
print(f"RBF SVM Accuracy: {svm_rbf.score(X_test_scaled, y_test):.3f}")

# Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3, coef0=1, C=1.0, random_state=42)
svm_poly.fit(X_train_scaled, y_train)

# Probability predictions (requires probability=True)
y_proba = svm_rbf.predict_proba(X_test_scaled)
print(f"Probability shape: {y_proba.shape}")

# Decision function values (distance from hyperplane)
decision_values = svm_rbf.decision_function(X_test_scaled)
print(f"Decision values shape: {decision_values.shape}")
```

**Custom Kernels:**

```python
# Define custom kernel
def custom_kernel(X, Y):
    """Example: Exponential kernel"""
    from sklearn.metrics.pairwise import euclidean_distances
    gamma = 1.0
    return np.exp(-gamma * euclidean_distances(X, Y))

svm_custom = SVC(kernel=custom_kernel)
svm_custom.fit(X_train_scaled, y_train)
```

**Hyperparameter Tuning:**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# Use best model
best_svm = grid_search.best_estimator_
print(f"Test accuracy: {best_svm.score(X_test_scaled, y_test):.3f}")
```

### 1.3 Decision Trees and Ensemble Methods

Decision trees partition feature space through recursive binary splits, creating interpretable if-then rules. Ensemble methods combine multiple trees to reduce overfitting and improve generalization.

#### 1.3.1 Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Decision Tree with hyperparameters to prevent overfitting
dt = DecisionTreeClassifier(
    criterion='gini',        # Impurity measure: 'gini' or 'entropy'
    max_depth=5,            # Maximum tree depth (None = unlimited)
    min_samples_split=20,   # Minimum samples to split an internal node
    min_samples_leaf=10,    # Minimum samples in a leaf node
    max_features='sqrt',    # Number of features to consider for splits
    random_state=42
)

dt.fit(X_train, y_train)

# Tree structure analysis
print(f"Tree depth: {dt.get_depth()}")
print(f"Number of leaves: {dt.get_n_leaves()}")
print(f"Feature importances:\n{dict(zip(iris.feature_names, dt.feature_importances_))}")

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, feature_names=iris.feature_names, 
          class_names=iris.target_names, rounded=True)
plt.savefig('decision_tree.png', dpi=100, bbox_inches='tight')

# Prediction path for interpretation
sample = X_test[0].reshape(1, -1)
decision_path = dt.decision_path(sample)
leaf_id = dt.apply(sample)
print(f"Sample reached leaf node: {leaf_id}")
```

**Cost Complexity Pruning:**

```python
from sklearn.model_selection import cross_val_score

# Get pruning path
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # Remove last alpha (trivial tree)

# Train trees with different alpha values
trees = []
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    tree.fit(X_train, y_train)
    trees.append(tree)

# Find optimal alpha via cross-validation
train_scores = [tree.score(X_train, y_train) for tree in trees]
test_scores = [tree.score(X_test, y_test) for tree in trees]

optimal_idx = np.argmax(test_scores)
optimal_alpha = ccp_alphas[optimal_idx]
print(f"Optimal CCP alpha: {optimal_alpha:.5f}")
```

#### 1.3.2 Random Forest

Random Forest builds multiple decision trees on bootstrapped samples with random feature subsets, aggregating predictions through majority voting (classification) or averaging (regression).

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=100,           # Number of trees
    max_depth=10,               # Maximum depth of each tree
    min_samples_split=10,       # Minimum samples to split
    min_samples_leaf=4,         # Minimum samples in leaf
    max_features='sqrt',        # Features per split: sqrt(n_features)
    bootstrap=True,             # Bootstrap sampling
    oob_score=True,             # Out-of-bag score estimation
    n_jobs=-1,                  # Parallel processing
    random_state=42
)

rf.fit(X_train, y_train)

# Performance metrics
print(f"OOB Score: {rf.oob_score_:.3f}")
print(f"Test Accuracy: {rf.score(X_test, y_test):.3f}")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': [f'feature_{i}' for i in range(X.shape[1])],
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10))

# Individual tree predictions
tree_predictions = np.array([tree.predict(X_test) for tree in rf.estimators_])
print(f"Predictions shape: {tree_predictions.shape}")  # (n_estimators, n_test_samples)

# Prediction confidence through voting
from scipy import stats
vote_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), 
                                  axis=0, arr=tree_predictions)
confidence = vote_counts.max(axis=0) / rf.n_estimators
print(f"Average prediction confidence: {confidence.mean():.3f}")
```

**Feature Selection with Random Forest:**

```python
from sklearn.feature_selection import SelectFromModel

# Select features with importance above threshold
selector = SelectFromModel(rf, threshold='median', prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print(f"Original features: {X_train.shape[1]}")
print(f"Selected features: {X_train_selected.shape[1]}")
print(f"Selected feature indices: {selector.get_support(indices=True)}")

# Train new model on selected features
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)
print(f"Test accuracy with selected features: {rf_selected.score(X_test_selected, y_test):.3f}")
```

#### 1.3.3 Gradient Boosting

Gradient Boosting sequentially builds weak learners (typically shallow trees) where each tree corrects errors of the previous ensemble, minimizing a differentiable loss function through gradient descent.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb = GradientBoostingClassifier(
    n_estimators=100,           # Number of boosting stages
    learning_rate=0.1,          # Shrinks contribution of each tree
    max_depth=3,                # Depth of each tree (keep shallow)
    min_samples_split=10,       # Minimum samples to split
    min_samples_leaf=4,         # Minimum samples in leaf
    subsample=0.8,              # Fraction of samples for each tree (stochastic boosting)
    max_features='sqrt',        # Features to consider for splits
    validation_fraction=0.1,    # Hold-out validation for early stopping
    n_iter_no_change=10,        # Early stopping rounds
    random_state=42
)

gb.fit(X_train, y_train)

print(f"Training score: {gb.score(X_train, y_train):.3f}")
print(f"Test score: {gb.score(X_test, y_test):.3f}")
print(f"Number of estimators used: {gb.n_estimators_}")

# Staged predictions for monitoring
import pandas as pd
train_scores = []
test_scores = []

for i, (train_pred, test_pred) in enumerate(zip(
    gb.staged_predict(X_train),
    gb.staged_predict(X_test)
)):
    train_scores.append(np.mean(train_pred == y_train))
    test_scores.append(np.mean(test_pred == y_test))

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_scores, label='Training Score')
plt.plot(test_scores, label='Test Score')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Gradient Boosting Learning Curves')
plt.savefig('gb_learning_curves.png')
```

**HistGradientBoosting (Fast Implementation):**

```python
from sklearn.ensemble import HistGradientBoostingClassifier

# Optimized for large datasets (native handling of missing values and categorical features)
hist_gb = HistGradientBoostingClassifier(
    max_iter=100,               # Number of boosting iterations
    learning_rate=0.1,
    max_depth=10,               # Maximum depth of trees
    min_samples_leaf=20,
    l2_regularization=0.1,      # Regularization parameter
    early_stopping=True,        # Automatic early stopping
    validation_fraction=0.1,
    random_state=42
)

hist_gb.fit(X_train, y_train)
print(f"HistGB Test Accuracy: {hist_gb.score(X_test, y_test):.3f}")

# Native categorical feature support
# X_with_categorical = X_train.copy()
# categorical_features = [False] * X.shape[1]  # Boolean mask
# hist_gb_cat = HistGradientBoostingClassifier(categorical_features=categorical_features)
```

### 1.4 K-Nearest Neighbors (KNN)

KNN is a non-parametric, instance-based learning algorithm that classifies samples based on majority voting among k nearest neighbors in feature space. It requires no training phase but incurs computational cost during prediction.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np

# Load digits dataset
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Feature scaling is crucial for distance-based algorithms
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(
    n_neighbors=5,              # Number of neighbors
    weights='uniform',          # 'uniform' or 'distance' (closer neighbors weighted more)
    algorithm='auto',           # 'auto', 'ball_tree', 'kd_tree', 'brute'
    metric='minkowski',         # Distance metric
    p=2,                        # Power parameter for Minkowski (p=2 is Euclidean)
    n_jobs=-1                   # Parallel processing
)

knn.fit(X_train_scaled, y_train)

print(f"Test Accuracy: {knn.score(X_test_scaled, y_test):.3f}")

# Get distances and indices of neighbors
distances, indices = knn.kneighbors(X_test_scaled[:5])
print(f"Distances shape: {distances.shape}")  # (5, n_neighbors)
print(f"Neighbor indices shape: {indices.shape}")

# Probability estimates (proportion of neighbors)
y_proba = knn.predict_proba(X_test_scaled[:5])
print(f"Probability predictions:\n{y_proba}")
```

**Finding Optimal K:**

```python
from sklearn.model_selection import cross_val_score

k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_temp, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal K: {optimal_k}")
print(f"Best CV Accuracy: {max(cv_scores):.3f}")

# Plot K vs Accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN: Finding Optimal K')
plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
plt.legend()
plt.grid(True)
```

**Distance Metrics Comparison:**

```python
metrics = ['euclidean', 'manhattan', 'cosine', 'chebyshev']
results = {}

for metric in metrics:
    if metric == 'cosine':
        knn_metric = KNeighborsClassifier(n_neighbors=5, metric=metric, algorithm='brute')
    else:
        knn_metric = KNeighborsClassifier(n_neighbors=5, metric=metric)
    
    knn_metric.fit(X_train_scaled, y_train)
    results[metric] = knn_metric.score(X_test_scaled, y_test)

print("Accuracy by Distance Metric:")
for metric, accuracy in results.items():
    print(f"{metric}: {accuracy:.3f}")
```

### 1.5 Naive Bayes

Naive Bayes classifiers apply Bayes' theorem with the "naive" assumption of feature independence. Despite this simplification, they perform surprisingly well in text classification, spam detection, and real-time prediction scenarios.

**Types:**
- **GaussianNB:** Features follow normal distribution (continuous data)
- **MultinomialNB:** Discrete counts (text, word frequencies)
- **BernoulliNB:** Binary features (document presence/absence)
- **ComplementNB:** Improved version of MultinomialNB for imbalanced datasets

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# Example 1: Gaussian Naive Bayes for continuous features
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

gnb = GaussianNB(
    priors=None,    # Prior probabilities (None = compute from data)
    var_smoothing=1e-9  # Variance smoothing for numerical stability
)

gnb.fit(X_train, y_train)
print(f"Gaussian NB Accuracy: {gnb.score(X_test, y_test):.3f}")

# Class statistics
print(f"Class priors: {gnb.class_prior_}")
print(f"Means shape: {gnb.theta_.shape}")  # (n_classes, n_features)
print(f"Variances shape: {gnb.var_.shape}")

# Example 2: Multinomial Naive Bayes for text classification
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, 
                                     remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,
                                    remove=('headers', 'footers', 'quotes'))

vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_train_counts = vectorizer.fit_transform(newsgroups_train.data)
X_test_counts = vectorizer.transform(newsgroups_test.data)

mnb = MultinomialNB(
    alpha=1.0,      # Additive (Laplace) smoothing parameter
    fit_prior=True  # Learn class prior probabilities
)

mnb.fit(X_train_counts, newsgroups_train.target)
print(f"\nMultinomial NB Accuracy: {mnb.score(X_test_counts, newsgroups_test.target):.3f}")

# Top features per class
feature_names = vectorizer.get_feature_names_out()
for i, category in enumerate(categories):
    top_indices = np.argsort(mnb.feature_log_prob_[i])[-10:]
    top_features = [feature_names[idx] for idx in top_indices]
    print(f"\n{category}: {top_features}")

# Example 3: Bernoulli Naive Bayes for binary features
bnb = BernoulliNB(alpha=1.0, binarize=0.0)  # Binarize threshold
bnb.fit(X_train_counts, newsgroups_train.target)
print(f"\nBernoulli NB Accuracy: {bnb.score(X_test_counts, newsgroups_test.target):.3f}")

# Example 4: Complement Naive Bayes for imbalanced datasets
cnb = ComplementNB(alpha=1.0, norm=True)
cnb.fit(X_train_counts, newsgroups_train.target)
print(f"Complement NB Accuracy: {cnb.score(X_test_counts, newsgroups_test.target):.3f}")
```

**Partial Fit for Incremental Learning:**

```python
# Useful for streaming data or when dataset doesn't fit in memory
from sklearn.naive_bayes import MultinomialNB
import numpy as np

mnb_incremental = MultinomialNB()

# Simulate streaming data in batches
batch_size = 100
n_samples = X_train_counts.shape[0]

for i in range(0, n_samples, batch_size):
    end_idx = min(i + batch_size, n_samples)
    X_batch = X_train_counts[i:end_idx]
    y_batch = newsgroups_train.target[i:end_idx]
    
    # Partial fit requires classes parameter on first call
    if i == 0:
        mnb_incremental.partial_fit(X_batch, y_batch, classes=np.unique(newsgroups_train.target))
    else:
        mnb_incremental.partial_fit(X_batch, y_batch)

print(f"Incremental NB Accuracy: {mnb_incremental.score(X_test_counts, newsgroups_test.target):.3f}")
```

### Common Pitfalls and Real-World Tips

**1. Feature Scaling:**
- **Critical for:** SVM, KNN, Logistic Regression with regularization
- **Not needed for:** Tree-based models (Decision Trees, Random Forest, Gradient Boosting)
- Always scale training and test data using the same fitted scaler

**2. Imbalanced Datasets:**
```python
# Technique 1: Class weights
log_reg_balanced = LogisticRegression(class_weight='balanced')

# Technique 2: Custom weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_dict = dict(enumerate(class_weights))

# Technique 3: Resampling (SMOTE, undersampling)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Technique 4: Appropriate metrics (F1, ROC-AUC instead of accuracy)
from sklearn.metrics import f1_score, roc_auc_score
f1 = f1_score(y_test, y_pred, average='weighted')
```

**3. Overfitting Prevention:**
- Use cross-validation instead of single train-test split
- Regularization in linear models (tune C parameter)
- Limit tree depth and minimum samples in tree-based models
- Early stopping in gradient boosting
- Reduce model complexity before gathering more data

**4. Curse of Dimensionality for KNN:**
- High-dimensional spaces make distance metrics less meaningful
- Apply dimensionality reduction (PCA, feature selection) before KNN
- Consider approximate nearest neighbors for large datasets

**5. SVM Computational Cost:**
- SVM training time: O(n²) to O(n³) for full data
- Use LinearSVC for large datasets (liblinear implementation)
- Consider SGDClassifier with hinge loss for very large datasets
- HistGradientBoosting often outperforms SVM with less tuning

**6. Probability Calibration:**
```python
from sklearn.calibration import CalibratedClassifierCV

# SVM and Naive Bayes often produce poorly calibrated probabilities
svm_calibrated = CalibratedClassifierCV(SVC(kernel='rbf'), cv=5)
svm_calibrated.fit(X_train, y_train)
calibrated_proba = svm_calibrated.predict_proba(X_test)
```

**7. Feature Engineering for Tree-Based Models:**
- Trees can't extrapolate beyond training range
- Create interaction features manually for important combinations
- Trees handle missing values, scaling, and non-linear relationships naturally

**8. Memory Efficiency:**
```python
# Use sparse matrices for high-dimensional sparse data (text)
from scipy.sparse import csr_matrix

# Reduce float precision
X_train_float32 = X_train.astype(np.float32)

# Use partial_fit for online learning
# Use warm_start=True in ensemble methods to add trees incrementally
```

### Interview Questions and Answers

**Q1: Explain the bias-variance tradeoff in the context of Random Forest vs single Decision Tree.**

**A:** A single decision tree has low bias (can model complex relationships) but high variance (overfits to training data, sensitive to small changes). Random Forest reduces variance through ensemble averaging while maintaining low bias. Each tree in the forest is trained on a bootstrap sample with random feature subsets, creating diverse but correlated predictions. When averaged, random errors cancel out while systematic patterns remain, reducing overfitting. The tradeoff: Random Forest sacrifices some interpretability and increases computational cost, but typically achieves 2-5% accuracy improvement over single trees on test data.

**Q2: When would you choose Logistic Regression over more complex models like Random Forest or Gradient Boosting?**

**A:** Choose Logistic Regression when:
- **Interpretability is critical:** Coefficients directly show feature impact and direction (e.g., credit scoring, medical diagnosis with regulatory requirements)
- **Linear relationships exist:** When decision boundaries are roughly linear, simpler models generalize better
- **High-dimensional sparse data:** Works well with text data after TF-IDF (faster training than trees)
- **Probability calibration matters:** Produces well-calibrated probabilities by default
- **Low latency required:** Prediction is O(n_features) vs O(n_trees × depth) for ensembles
- **Limited training data:** Less prone to overfitting than complex models with small datasets (<1000 samples)

**Q3: How does the C parameter affect SVM, and how would you tune it?**

**A:** The C parameter controls the regularization strength (inverse of lambda). 
- **Small C (strong regularization):** Wider margin, allows more misclassifications, simpler decision boundary (high bias, low variance)
- **Large C (weak regularization):** Narrow margin, penalizes misclassifications heavily, complex boundary (low bias, high variance)

**Tuning approach:**
```python
# Use logarithmic scale for grid search
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]}

# Start with wide range, then narrow down
# Monitor training vs validation scores for overfitting
# For imbalanced data, optimize F1 instead of accuracy
```

**Q4: Why is feature scaling essential for some algorithms but not others?**

**A:** 
**Requires scaling:** Distance-based (KNN, SVM), gradient-based (Logistic Regression, Neural Networks), L1/L2 regularized models
- Reason: Features with larger scales dominate distance calculations or gradient updates. A feature in [0, 1000] overwhelms one in [0, 1].

**Doesn't require scaling:** Tree-based (Decision Trees, Random Forest, Gradient Boosting), Naive Bayes
- Reason: Trees make split decisions based on feature thresholds relative to that feature alone, not across features. Naive Bayes uses probability distributions within each feature.

**Q5: Explain how Gradient Boosting differs from Random Forest in terms of bias-variance tradeoff.**

**A:**
- **Random Forest:** Reduces variance through averaging independent deep trees (bagging). Each tree has low bias, high variance. Final model: low bias, low variance.
- **Gradient Boosting:** Reduces bias through sequential correction with shallow trees (boosting). Each tree has high bias, low variance. Iteratively fits residuals, focusing on hard examples.

**Key differences:**
- RF trains in parallel; GB trains sequentially
- RF uses deep trees (high variance); GB uses shallow trees (high bias)
- GB more prone to overfitting (requires early stopping, learning rate tuning)
- GB typically achieves higher accuracy but requires more careful tuning
- RF more robust to hyperparameters

**Q6: How would you handle a multi-class classification problem with 100+ classes?**

**A:** Strategies:
1. **Hierarchical classification:** Group similar classes, predict at coarse level first, then fine-grained
2. **One-vs-Rest decomposition:** Train binary classifier per class (default in sklearn)
3. **Error-correcting output codes:** Encode classes as binary codes, train multiple binary classifiers
4. **Optimize for speed:**
   - Use LinearSVC or SGDClassifier instead of SVC
   - HistGradientBoosting with early stopping
   - Reduce dimensionality before classification (PCA, feature selection)
5. **Probability calibration:** Use CalibratedClassifierCV for reliable confidence scores
6. **Evaluation:** Micro/macro-averaged F1, confusion matrix analysis, per-class metrics

**Q7: In production, your SVM model takes too long to predict. What optimization strategies would you use?**

**A:** 
1. **Switch to linear kernel:** Replace RBF with LinearSVC (10-100× faster)
2. **Reduce support vectors:** Use higher C to create sparser model
3. **Approximate methods:** Use SGDClassifier with hinge loss (linear SVM approximation)
4. **Dimensionality reduction:** PCA or feature selection before SVM
5. **Model replacement:** Consider Logistic Regression or HistGradientBoosting (often comparable accuracy, much faster)
6. **Batch predictions:** Use predict() instead of individual predictions
7. **Quantization:** Convert model to lower precision (float32 or int8)
8. **Hardware acceleration:** GPU-based inference (cuML library)

**Q8: Explain the difference between `predict()` and `predict_proba()`. When would you use `decision_function()`?**

**A:**
- **`predict()`**: Returns hard class labels (argmax of probabilities)
  - Use when: You need definitive classifications
  
- **`predict_proba()`**: Returns probability estimates for each class
  - Use when: You need confidence scores, want to set custom thresholds, need calibrated probabilities for risk assessment
  - Note: Some classifiers (SVM, SGD) require `probability=True` during training
  
- **`decision_function()`**: Returns raw decision scores (distance from hyperplane for SVM, log-odds for others)
  - Use when: You need ranking without probability interpretation, comparing scores across models, threshold optimization
  - Faster than predict_proba (no probability calibration overhead)

```python
# Example: Custom threshold for imbalanced classification
scores = model.decision_function(X_test)
custom_threshold = -0.5  # Favor recall over precision
y_pred_custom = (scores >= custom_threshold).astype(int)
```

**Q9: Your Random Forest model has 95% training accuracy but only 70% test accuracy. What steps would you take?**

**A:** This indicates overfitting. Solutions:
1. **Reduce model complexity:**
   - Decrease `max_depth` (try 5-10 instead of None)
   - Increase `min_samples_split` and `min_samples_leaf` (e.g., 20-50)
   - Reduce `n_estimators` (though less impactful)
   - Set `max_features='sqrt'` to increase randomness

2. **Cross-validation:** Use 5-10 fold CV to get reliable performance estimates

3. **Data-centric approaches:**
   - Collect more training data if possible
   - Check for data leakage (future information in training)
   - Ensure train-test distribution similarity
   - Remove outliers or noisy labels

4. **Feature engineering:**
   - Remove high-cardinality features (e.g., user IDs)
   - Remove highly correlated features
   - Apply feature selection (SelectFromModel)

5. **Alternative models:** Try simpler models (Logistic Regression) or less prone to overfitting (Ridge Classifier)

**Q10: How do you handle categorical features in tree-based models vs SVM/Logistic Regression?**

**A:**
**Tree-based models (native handling):**
```python
# HistGradientBoosting supports categorical features directly
from sklearn.ensemble import HistGradientBoostingClassifier
cat_features = [True, False, True, False]  # Boolean mask
model = HistGradientBoostingClassifier(categorical_features=cat_features)

# For other tree models, use ordinal encoding (order doesn't matter for trees)
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X_categorical)
```

**Distance/Linear models (require encoding):**
```python
# One-hot encoding (preferred for low-cardinality features)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_ohe = ohe.fit_transform(X_categorical)

# Target encoding for high-cardinality features
from category_encoders import TargetEncoder
te = TargetEncoder()
X_encoded = te.fit_transform(X_categorical, y)

# Never use label encoding (0, 1, 2...) as it implies ordinal relationship
```

---

## 2. Regression

### Definition and Fundamentals

Regression is a supervised learning task that predicts continuous numerical values based on input features. Scikit-learn provides a comprehensive suite of regression algorithms ranging from linear models (interpretable, efficient) to non-linear methods (flexible, complex). The goal is to learn a function f: X → y that minimizes prediction error on unseen data, balancing model complexity with generalization performance.

### 2.1 Linear Regression

Linear Regression models the relationship between features and target as a linear combination: y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ. It minimizes the mean squared error (MSE) using ordinary least squares (OLS), providing closed-form solutions for small to medium datasets.

**Mathematical Foundation:**
- **Cost function:** MSE = (1/n) Σ(yᵢ - ŷᵢ)²
- **Normal equation:** w = (XᵀX)⁻¹Xᵀy
- **Assumptions:** Linearity, independence, homoscedasticity, normality of residuals

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

# Generate synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=8,
                       noise=20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression
lr = LinearRegression(
    fit_intercept=True,     # Calculate intercept (bias term)
    copy_X=True,            # Copy X to avoid modification
    n_jobs=-1               # Parallel processing for multi-output regression
)

lr.fit(X_train, y_train)

# Model parameters
print(f"Coefficients: {lr.coef_}")
print(f"Intercept: {lr.intercept_}")
print(f"Number of features: {lr.n_features_in_}")

# Predictions
y_pred = lr.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nPerformance Metrics:")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R² Score: {r2:.3f}")

# Residual analysis
residuals = y_test - y_pred
print(f"\nResidual Statistics:")
print(f"Mean: {residuals.mean():.3f}")
print(f"Std: {residuals.std():.3f}")
print(f"Min: {residuals.min():.3f}")
print(f"Max: {residuals.max():.3f}")
```

**Feature Importance via Coefficients:**

```python
# For standardized features, coefficients represent relative importance
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_scaled = LinearRegression()
lr_scaled.fit(X_train_scaled, y_train)

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'feature': [f'feature_{i}' for i in range(X.shape[1])],
    'coefficient': lr_scaled.coef_,
    'abs_coefficient': np.abs(lr_scaled.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\nFeature Importance (Standardized Coefficients):")
print(feature_importance)
```

**Polynomial Features for Non-linear Relationships:**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create polynomial features and fit linear model
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

poly_pipeline.fit(X_train, y_train)
y_pred_poly = poly_pipeline.predict(X_test)

print(f"\nPolynomial Regression R² Score: {r2_score(y_test, y_pred_poly):.3f}")
print(f"Number of features after polynomial expansion: {poly_pipeline.named_steps['poly'].n_output_features_}")
```

### 2.2 Ridge Regression (L2 Regularization)

Ridge Regression adds L2 penalty (squared magnitude of coefficients) to the cost function, preventing overfitting by shrinking coefficients toward zero without eliminating features entirely. It's particularly effective for multicollinear features and high-dimensional data.

**Cost Function:** MSE + α × Σ(wᵢ²)

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

# Load diabetes dataset (real-world medical data)
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Feature scaling is crucial for regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge with fixed alpha
ridge = Ridge(
    alpha=1.0,              # Regularization strength (higher = more regularization)
    fit_intercept=True,
    solver='auto',          # 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
    max_iter=1000,
    random_state=42
)

ridge.fit(X_train_scaled, y_train)

print(f"Ridge R² Score: {ridge.score(X_test_scaled, y_test):.3f}")
print(f"Ridge Coefficients L2 Norm: {np.linalg.norm(ridge.coef_):.3f}")

# Compare with Linear Regression
lr_compare = LinearRegression()
lr_compare.fit(X_train_scaled, y_train)
print(f"Linear Regression R² Score: {lr_compare.score(X_test_scaled, y_test):.3f}")
print(f"Linear Regression Coefficients L2 Norm: {np.linalg.norm(lr_compare.coef_):.3f}")
```

**Hyperparameter Tuning with RidgeCV:**

```python
# RidgeCV: Efficient cross-validated Ridge with built-in alpha selection
alphas = np.logspace(-3, 3, 50)  # 50 alpha values from 0.001 to 1000

ridge_cv = RidgeCV(
    alphas=alphas,
    cv=5,                   # 5-fold cross-validation
    scoring='r2',           # Can also use 'neg_mean_squared_error'
    store_cv_values=True    # Store all CV scores
)

ridge_cv.fit(X_train_scaled, y_train)

print(f"\nOptimal alpha: {ridge_cv.alpha_:.4f}")
print(f"Best CV R² Score: {ridge_cv.best_score_:.3f}")
print(f"Test R² Score: {ridge_cv.score(X_test_scaled, y_test):.3f}")

# Visualize alpha selection
cv_mse_per_alpha = ridge_cv.cv_values_.mean(axis=0)
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, cv_mse_per_alpha)
plt.xlabel('Alpha')
plt.ylabel('Mean CV MSE')
plt.title('Ridge Regression: Alpha vs Cross-Validation MSE')
plt.axvline(ridge_cv.alpha_, color='r', linestyle='--', label=f'Optimal α={ridge_cv.alpha_:.4f}')
plt.legend()
plt.grid(True)
```

**Regularization Path Visualization:**

```python
# Show how coefficients shrink with increasing alpha
alphas_range = np.logspace(-2, 2, 100)
coefs = []

for alpha in alphas_range:
    ridge_temp = Ridge(alpha=alpha)
    ridge_temp.fit(X_train_scaled, y_train)
    coefs.append(ridge_temp.coef_)

coefs = np.array(coefs)

plt.figure(figsize=(12, 6))
for i in range(coefs.shape[1]):
    plt.plot(alphas_range, coefs[:, i], label=f'Feature {i}')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regularization Path')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
```

### 2.3 Lasso Regression (L1 Regularization)

Lasso (Least Absolute Shrinkage and Selection Operator) uses L1 penalty, which can drive coefficients exactly to zero, effectively performing automatic feature selection. It's ideal when you suspect only a subset of features are relevant.

**Cost Function:** MSE + α × Σ|wᵢ|

```python
from sklearn.linear_model import Lasso, LassoCV

# Lasso with fixed alpha
lasso = Lasso(
    alpha=1.0,
    fit_intercept=True,
    precompute=False,       # Precompute Gram matrix for speed
    max_iter=1000,
    tol=1e-4,               # Tolerance for optimization
    warm_start=False,       # Reuse previous solution for different alpha
    positive=False,         # Force positive coefficients
    selection='cyclic',     # 'cyclic' or 'random' feature selection order
    random_state=42
)

lasso.fit(X_train_scaled, y_train)

print(f"Lasso R² Score: {lasso.score(X_test_scaled, y_test):.3f}")
print(f"Number of non-zero coefficients: {np.sum(lasso.coef_ != 0)}")
print(f"Lasso Coefficients L1 Norm: {np.linalg.norm(lasso.coef_, ord=1):.3f}")

# Feature selection via Lasso
selected_features = np.where(lasso.coef_ != 0)[0]
print(f"Selected feature indices: {selected_features}")
```

**Cross-Validated Lasso with LassoCV:**

```python
lasso_cv = LassoCV(
    alphas=None,            # Auto-generate alpha values
    cv=5,
    max_iter=10000,
    n_jobs=-1,
    random_state=42
)

lasso_cv.fit(X_train_scaled, y_train)

print(f"\nOptimal alpha: {lasso_cv.alpha_:.4f}")
print(f"Test R² Score: {lasso_cv.score(X_test_scaled, y_test):.3f}")
print(f"Number of non-zero coefficients: {np.sum(lasso_cv.coef_ != 0)}")

# Mean squared error path
mse_path = lasso_cv.mse_path_.mean(axis=1)
alphas_path = lasso_cv.alphas_

plt.figure(figsize=(10, 6))
plt.semilogx(alphas_path, mse_path)
plt.xlabel('Alpha')
plt.ylabel('Mean CV MSE')
plt.title('Lasso: Alpha vs Cross-Validation MSE')
plt.axvline(lasso_cv.alpha_, color='r', linestyle='--', label=f'Optimal α={lasso_cv.alpha_:.4f}')
plt.legend()
plt.grid(True)
```

**Feature Selection Pipeline:**

```python
from sklearn.feature_selection import SelectFromModel

# Use Lasso for feature selection, then train different model
lasso_selector = SelectFromModel(
    Lasso(alpha=0.1, random_state=42),
    threshold='median'      # Select features with importance > median
)

lasso_selector.fit(X_train_scaled, y_train)
X_train_selected = lasso_selector.transform(X_train_scaled)
X_test_selected = lasso_selector.transform(X_test_scaled)

print(f"\nOriginal features: {X_train_scaled.shape[1]}")
print(f"Selected features: {X_train_selected.shape[1]}")

# Train model on selected features
lr_selected = LinearRegression()
lr_selected.fit(X_train_selected, y_train)
print(f"R² with selected features: {lr_selected.score(X_test_selected, y_test):.3f}")
```

### 2.4 ElasticNet (L1 + L2 Regularization)

ElasticNet combines L1 and L2 penalties, balancing Lasso's feature selection with Ridge's coefficient shrinkage. It's particularly useful when features are correlated—Lasso might arbitrarily select one, while ElasticNet tends to group them.

**Cost Function:** MSE + α × (l1_ratio × Σ|wᵢ| + (1-l1_ratio)/2 × Σwᵢ²)

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

elastic = ElasticNet(
    alpha=1.0,              # Overall regularization strength
    l1_ratio=0.5,           # L1 vs L2 balance: 0=Ridge, 1=Lasso, 0.5=equal mix
    fit_intercept=True,
    max_iter=1000,
    tol=1e-4,
    random_state=42
)

elastic.fit(X_train_scaled, y_train)

print(f"ElasticNet R² Score: {elastic.score(X_test_scaled, y_test):.3f}")
print(f"Number of non-zero coefficients: {np.sum(elastic.coef_ != 0)}")

# Cross-validated ElasticNet
elastic_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],  # Grid of L1 ratios
    alphas=None,            # Auto-generate alpha values
    cv=5,
    max_iter=10000,
    n_jobs=-1,
    random_state=42
)

elastic_cv.fit(X_train_scaled, y_train)

print(f"\nOptimal alpha: {elastic_cv.alpha_:.4f}")
print(f"Optimal l1_ratio: {elastic_cv.l1_ratio_:.2f}")
print(f"Test R² Score: {elastic_cv.score(X_test_scaled, y_test):.3f}")
```

**Comparing Regularization Methods:**

```python
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 50)),
    'Lasso': LassoCV(max_iter=10000),
    'ElasticNet': ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], max_iter=10000)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    r2_train = model.score(X_train_scaled, y_train)
    r2_test = model.score(X_test_scaled, y_test)
    
    # Count non-zero coefficients
    if hasattr(model, 'coef_'):
        n_nonzero = np.sum(model.coef_ != 0)
    else:
        n_nonzero = X_train_scaled.shape[1]
    
    results.append({
        'Model': name,
        'Train R²': r2_train,
        'Test R²': r2_test,
        'Non-zero Coefs': n_nonzero
    })

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))
```

### 2.5 Stochastic Gradient Descent (SGD) Regressor

SGDRegressor optimizes linear models using stochastic gradient descent, making it suitable for large-scale datasets that don't fit in memory. It supports various loss functions and regularization penalties.

```python
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(
    loss='squared_error',   # 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'
    penalty='l2',           # 'l1', 'l2', 'elasticnet', None
    alpha=0.0001,           # Regularization strength
    l1_ratio=0.15,          # ElasticNet mixing parameter
    max_iter=1000,
    tol=1e-3,
    learning_rate='invscaling',  # 'constant', 'optimal', 'invscaling', 'adaptive'
    eta0=0.01,              # Initial learning rate
    power_t=0.25,           # Exponent for inverse scaling
    early_stopping=True,    # Stop if validation score doesn't improve
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=42
)

sgd.fit(X_train_scaled, y_train)

print(f"SGD R² Score: {sgd.score(X_test_scaled, y_test):.3f}")
print(f"Number of iterations: {sgd.n_iter_}")
print(f"Intercept: {sgd.intercept_[0]:.3f}")
```

**Partial Fit for Online Learning:**

```python
# Simulate streaming data
sgd_online = SGDRegressor(random_state=42)

batch_size = 50
n_samples = X_train_scaled.shape[0]

for i in range(0, n_samples, batch_size):
    end_idx = min(i + batch_size, n_samples)
    X_batch = X_train_scaled[i:end_idx]
    y_batch = y_train[i:end_idx]
    
    sgd_online.partial_fit(X_batch, y_batch)
    
    if (i // batch_size) % 5 == 0:
        r2 = sgd_online.score(X_test_scaled, y_test)
        print(f"Batch {i//batch_size}: Test R² = {r2:.3f}")
```

### 2.6 Tree-Based Regression

Decision trees and ensemble methods (Random Forest, Gradient Boosting) handle non-linear relationships, interactions, and missing values without feature scaling.

#### 2.6.1 Decision Tree Regressor

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree

dt_reg = DecisionTreeRegressor(
    criterion='squared_error',  # 'squared_error', 'friedman_mse', 'absolute_error', 'poisson'
    splitter='best',            # 'best' or 'random'
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features=None,          # Consider all features
    random_state=42
)

dt_reg.fit(X_train, y_train)

print(f"Decision Tree R² Score: {dt_reg.score(X_test, y_test):.3f}")
print(f"Tree depth: {dt_reg.get_depth()}")
print(f"Number of leaves: {dt_reg.get_n_leaves()}")

# Feature importance
importance_df = pd.DataFrame({
    'feature': [f'feature_{i}' for i in range(X.shape[1])],
    'importance': dt_reg.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importances:")
print(importance_df.head())
```

#### 2.6.2 Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',        # 'sqrt', 'log2', or float
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

rf_reg.fit(X_train, y_train)

print(f"Random Forest OOB Score: {rf_reg.oob_score_:.3f}")
print(f"Random Forest R² Score: {rf_reg.score(X_test, y_test):.3f}")

# Prediction intervals using quantile forests approach
tree_predictions = np.array([tree.predict(X_test) for tree in rf_reg.estimators_])
predictions_mean = tree_predictions.mean(axis=0)
predictions_std = tree_predictions.std(axis=0)

# 95% confidence interval
lower_bound = predictions_mean - 1.96 * predictions_std
upper_bound = predictions_mean + 1.96 * predictions_std

print(f"\nAverage prediction interval width: {(upper_bound - lower_bound).mean():.3f}")
```

#### 2.6.3 Gradient Boosting Regressor

```python
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

gb_reg = GradientBoostingRegressor(
    loss='squared_error',       # 'squared_error', 'absolute_error', 'huber', 'quantile'
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,              # Stochastic gradient boosting
    criterion='friedman_mse',   # 'friedman_mse', 'squared_error'
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    validation_fraction=0.1,
    n_iter_no_change=10,        # Early stopping
    random_state=42
)

gb_reg.fit(X_train, y_train)

print(f"Gradient Boosting R² Score: {gb_reg.score(X_test, y_test):.3f}")
print(f"Number of estimators used: {gb_reg.n_estimators_}")

# Learning curves
train_scores = []
test_scores = []

for i, (train_pred, test_pred) in enumerate(zip(
    gb_reg.staged_predict(X_train),
    gb_reg.staged_predict(X_test)
)):
    train_scores.append(r2_score(y_train, train_pred))
    test_scores.append(r2_score(y_test, test_pred))

plt.figure(figsize=(10, 6))
plt.plot(train_scores, label='Training R²')
plt.plot(test_scores, label='Test R²')
plt.xlabel('Number of Estimators')
plt.ylabel('R² Score')
plt.title('Gradient Boosting Learning Curves')
plt.legend()
plt.grid(True)
```

**HistGradientBoosting for Large Datasets:**

```python
# Optimized implementation with native support for missing values and categorical features
hist_gb_reg = HistGradientBoostingRegressor(
    loss='squared_error',
    learning_rate=0.1,
    max_iter=100,
    max_depth=10,
    min_samples_leaf=20,
    l2_regularization=0.1,
    max_bins=255,               # Number of bins for feature discretization
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)

hist_gb_reg.fit(X_train, y_train)

print(f"\nHistGradientBoosting R² Score: {hist_gb_reg.score(X_test, y_test):.3f}")
print(f"Number of iterations: {hist_gb_reg.n_iter_}")
```

### 2.7 Support Vector Regression (SVR)

SVR applies SVM principles to regression by fitting data within an epsilon-tube, tolerating errors within this margin. It's effective for non-linear regression but computationally expensive for large datasets.

```python
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import StandardScaler

# Feature scaling is CRITICAL for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RBF kernel SVR
svr = SVR(
    kernel='rbf',
    C=1.0,                  # Regularization parameter
    epsilon=0.1,            # Epsilon-tube width
    gamma='scale',          # Kernel coefficient
    cache_size=200,         # Kernel cache size in MB
    max_iter=-1             # No limit
)

svr.fit(X_train_scaled, y_train)

print(f"SVR R² Score: {svr.score(X_test_scaled, y_test):.3f}")
print(f"Number of support vectors: {len(svr.support_)}")

# Linear SVR (much faster for large datasets)
linear_svr = LinearSVR(
    epsilon=0.0,
    C=1.0,
    loss='epsilon_insensitive',  # 'epsilon_insensitive' or 'squared_epsilon_insensitive'
    max_iter=1000,
    random_state=42
)

linear_svr.fit(X_train_scaled, y_train)
print(f"Linear SVR R² Score: {linear_svr.score(X_test_scaled, y_test):.3f}")
```

**Hyperparameter Tuning for SVR:**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1.0],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(
    SVR(kernel='rbf'),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV R² Score: {grid_search.best_score_:.3f}")
print(f"Test R² Score: {grid_search.best_estimator_.score(X_test_scaled, y_test):.3f}")
```

### 2.8 K-Nearest Neighbors Regressor

KNN Regressor predicts by averaging the target values of k nearest neighbors. It's simple and non-parametric but sensitive to feature scaling and curse of dimensionality.

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_reg = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',         # 'uniform' or 'distance' (inverse distance weighting)
    algorithm='auto',           # 'auto', 'ball_tree', 'kd_tree', 'brute'
    metric='minkowski',
    p=2,                        # Euclidean distance
    n_jobs=-1
)

knn_reg.fit(X_train_scaled, y_train)

print(f"KNN Regressor R² Score: {knn_reg.score(X_test_scaled, y_test):.3f}")

# Find optimal K
from sklearn.model_selection import cross_val_score

k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn_temp = KNeighborsRegressor(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn_temp, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print(f"\nOptimal K: {optimal_k}")
print(f"Best CV R² Score: {max(cv_scores):.3f}")
```

### Common Pitfalls and Real-World Tips

**1. Multicollinearity Detection:**
```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Calculate Variance Inflation Factor (VIF)
def calculate_vif(X):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
    
    return vif_data.sort_values('VIF', ascending=False)

# VIF > 10 indicates severe multicollinearity
# Solution: Use Ridge/ElasticNet instead of Lasso, or perform PCA
```

**2. Residual Analysis for Model Diagnostics:**
```python
import scipy.stats as stats

def diagnose_residuals(y_true, y_pred):
    """Check linear regression assumptions"""
    residuals = y_true - y_pred
    
    # 1. Normality test (Shapiro-Wilk)
    stat, p_value = stats.shapiro(residuals)
    print(f"Normality test p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("⚠️ Residuals are not normally distributed")
    
    # 2. Homoscedasticity (constant variance)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    # 3. Q-Q plot for normality
    plt.subplot(132)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    # 4. Histogram of residuals
    plt.subplot(133)
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig('residual_analysis.png')

# Usage
y_pred = lr.predict(X_test)
diagnose_residuals(y_test, y_pred)
```

**3. Handling Outliers:**
```python
from sklearn.linear_model import HuberRegressor, RANSACRegressor

# Huber Regressor: Robust to outliers using Huber loss
huber = HuberRegressor(
    epsilon=1.35,           # Controls outlier threshold
    max_iter=100,
    alpha=0.0001
)
huber.fit(X_train_scaled, y_train)

# RANSAC: Random Sample Consensus (fits on inliers only)
ransac = RANSACRegressor(
    estimator=LinearRegression(),
    min_samples=50,         # Minimum samples for fitting
    residual_threshold=None, # Auto-calculate based on MAD
    max_trials=100,
    random_state=42
)
ransac.fit(X_train, y_train)

# Identify inliers and outliers
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
print(f"Inliers: {inlier_mask.sum()}, Outliers: {outlier_mask.sum()}")
```

**4. Feature Scaling Impact:**
```python
# Demonstration: Feature scaling necessity
models_scaling_sensitive = {
    'Ridge (no scaling)': Ridge(alpha=1.0),
    'Ridge (with scaling)': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
    'Random Forest (no scaling)': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models_scaling_sensitive.items():
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    print(f"{name}: R² = {r2:.3f}")

# Result: Ridge needs scaling, RF doesn't
```

**5. Dealing with Non-Linear Relationships:**
```python
# Strategy 1: Polynomial features + regularization
from sklearn.preprocessing import PolynomialFeatures

poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 50)))
])

# Strategy 2: Spline transformations
from sklearn.preprocessing import SplineTransformer

spline_pipeline = Pipeline([
    ('spline', SplineTransformer(n_knots=5, degree=3)),
    ('ridge', Ridge(alpha=1.0))
])

# Strategy 3: Tree-based models (handle non-linearity naturally)
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5)
```

**6. Cross-Validation Strategies:**
```python
from sklearn.model_selection import KFold, cross_val_score, cross_validate

# Standard K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Multiple metrics evaluation
scoring = {
    'r2': 'r2',
    'neg_mse': 'neg_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error'
}

cv_results = cross_validate(
    Ridge(alpha=1.0),
    X_train_scaled,
    y_train,
    cv=kf,
    scoring=scoring,
    return_train_score=True
)

print(f"Test R²: {cv_results['test_r2'].mean():.3f} ± {cv_results['test_r2'].std():.3f}")
print(f"Test MAE: {-cv_results['test_neg_mae'].mean():.3f}")
print(f"Train-Test R² Gap: {cv_results['train_r2'].mean() - cv_results['test_r2'].mean():.3f}")
```

**7. Memory-Efficient Practices:**
```python
# For large datasets, use:
# 1. SGDRegressor with partial_fit
# 2. HistGradientBoostingRegressor (faster than GradientBoostingRegressor)
# 3. LinearSVR instead of SVR
# 4. Reduce float precision

X_train_float32 = X_train.astype(np.float32)
y_train_float32 = y_train.astype(np.float32)

# 5. Use sparse matrices for high-dimensional sparse data
from scipy.sparse import csr_matrix
X_sparse = csr_matrix(X_train)
```

**8. Handling Categorical Variables:**
```python
# Never use label encoding for nominal categories in linear models
# Use one-hot encoding for linear models, target encoding for trees

from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

# For linear models
ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')

# For tree-based models (high cardinality)
te = TargetEncoder(smoothing=1.0)  # Smoothing prevents overfitting
```

### Interview Questions and Answers

**Q1: Explain the difference between R² and adjusted R². When would adjusted R² decrease while R² increases?**

**A:** 
- **R²** measures the proportion of variance in the target explained by features: R² = 1 - (SS_residual / SS_total). It always increases (or stays constant) when adding features, even irrelevant ones.
- **Adjusted R²** penalizes model complexity: Adj R² = 1 - [(1-R²)(n-1)/(n-p-1)], where n=samples, p=features.

**When Adj R² decreases while R² increases:** When adding features that contribute less to explained variance than they cost in complexity. Example: Adding 10 random noise features to a model with 100 samples might increase R² from 0.75 to 0.76 but decrease Adj R² from 0.73 to 0.68.

**Production implication:** Use adjusted R² or cross-validation for model selection, not raw R².

**Q2: Your Ridge regression model has similar training and test R² (both ~0.65), but stakeholders want better predictions. What would you try?**

**A:** Similar train/test scores suggest **high bias (underfitting)**, not overfitting. Solutions:

1. **Feature engineering:**
   - Create polynomial/interaction features
   - Domain-specific transformations (log, sqrt for skewed distributions)
   - Binning continuous variables
   - Extract date features (day_of_week, month, is_holiday)

2. **Switch to non-linear models:**
   - Gradient Boosting (captures complex interactions)
   - Random Forest
   - Neural networks

3. **Reduce regularization:** Decrease alpha in Ridge (currently over-regularized)

4. **Collect more informative features:** Current features may have low predictive power

5. **Check for data quality issues:** Outliers, measurement errors, label noise

**Don't:** Add more data (helps with overfitting, not underfitting) or increase model complexity through more regularization.

**Q3: How do you choose between Ridge, Lasso, and ElasticNet in production?**

**A:** Decision framework:

**Use Lasso when:**
- Feature selection is important (automatic elimination)
- High-dimensional data with many irrelevant features (p >> n)
- Interpretability is critical (sparse solutions)
- Example: Genomics (thousands of genes, few relevant)

**Use Ridge when:**
- Most features are relevant
- Features are highly correlated (Lasso arbitrarily picks one; Ridge shrinks all)
- Prediction accuracy > interpretability
- Example: Image regression (all pixels contribute)

**Use ElasticNet when:**
- Features are correlated AND you want some feature selection
- Lasso is unstable (selects different features on similar datasets)
- Middle ground between Ridge and Lasso
- Example: Financial modeling with correlated economic indicators

**Production workflow:**
```python
# Use ElasticNetCV to automatically select l1_ratio
elastic_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5)
elastic_cv.fit(X_train, y_train)
# If optimal l1_ratio ≈ 1 → use Lasso
# If optimal l1_ratio ≈ 0 → use Ridge
```

**Q4: Explain why Random Forest typically outperforms a single Decision Tree, and when it might not.**

**A:**
**Why RF outperforms:**
- **Variance reduction:** Averages predictions from multiple trees trained on different bootstrap samples, reducing overfitting
- **Feature randomness:** Each split considers random subset of features, decorrelating trees
- **Robustness:** Less sensitive to outliers, missing values, and hyperparameters
- **Example:** Single tree might achieve 0.6 test R², RF with 100 trees achieves 0.75-0.78 R²

**When RF might underperform:**
- **Extrapolation tasks:** Trees can't predict beyond training range. Linear models extrapolate better.
- **High-dimensional sparse data:** Computational cost and memory (text data with 100K features)
- **Smooth linear relationships:** Linear regression is faster, more interpretable, equally accurate
- **Very small datasets:** (<100 samples) RF's complexity leads to overfitting; simple models better
- **Real-time low-latency requirements:** RF prediction slower than linear models

**Q5: Your Gradient Boosting model has 0.95 training R² but 0.72 test R². How would you address this overfitting?**

**A:** Clear overfitting (high variance). Prioritized solutions:

1. **Reduce model complexity:**
   ```python
   # Decrease tree depth (most effective)
   max_depth=3  # instead of 5+
   
   # Increase minimum samples
   min_samples_split=50
   min_samples_leaf=20
   
   # Reduce number of estimators
   n_estimators=50  # instead of 200+
   ```

2. **Increase regularization:**
   ```python
   # Lower learning rate + more trees with early stopping
   learning_rate=0.01  # instead of 0.1
   n_estimators=500
   n_iter_no_change=10
   
   # Add L2 regularization (HistGradientBoosting)
   l2_regularization=1.0
   
   # Use subsampling
   subsample=0.7
   max_features='sqrt'
   ```

3. **Data augmentation:**
   - Collect more training data
   - Use cross-validation to detect overfitting early
   - Check for data leakage (future information in features)

4. **Switch to Random Forest:** More robust to overfitting, requires less tuning

**Q6: How would you implement a regression model that provides prediction intervals, not just point estimates?**

**A:** Multiple approaches:

**1. Quantile Regression (most direct):**
```python
from sklearn.ensemble import GradientBoostingRegressor

# Train multiple models for different quantiles
quantiles = [0.1, 0.5, 0.9]  # 10th, 50th (median), 90th percentiles
models = {}

for q in quantiles:
    gb = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=100)
    gb.fit(X_train, y_train)
    models[q] = gb

# Predictions with 80% interval
lower = models[0.1].predict(X_test)
median = models[0.5].predict(X_test)
upper = models[0.9].predict(X_test)
```

**2. Bootstrap Confidence Intervals:**
```python
from sklearn.utils import resample

predictions = []
for _ in range(100):
    X_boot, y_boot = resample(X_train, y_train, random_state=None)
    model = Ridge(alpha=1.0)
    model.fit(X_boot, y_boot)
    predictions.append(model.predict(X_test))

predictions = np.array(predictions)
lower = np.percentile(predictions, 2.5, axis=0)
upper = np.percentile(predictions, 97.5, axis=0)
```

**3. Random Forest Standard Deviation:**
```python
# Use individual tree predictions
tree_preds = np.array([tree.predict(X_test) for tree in rf.estimators_])
mean_pred = tree_preds.mean(axis=0)
std_pred = tree_preds.std(axis=0)

# Approximate 95% interval
lower = mean_pred - 1.96 * std_pred
upper = mean_pred + 1.96 * std_pred
```

**4. Conformal Prediction (distribution-free):**
```python
# Calculate residuals on calibration set
cal_residuals = np.abs(y_cal - model.predict(X_cal))
quantile_val = np.quantile(cal_residuals, 0.95)

# Prediction intervals
predictions = model.predict(X_test)
lower = predictions - quantile_val
upper = predictions + quantile_val
```

**Q7: You have 1 million samples with 50 features. Which regression algorithms would be practical, and which should you avoid?**

**A:**
**Practical (fast training/prediction):**
- **SGDRegressor:** O(n) complexity, memory-efficient, supports partial_fit
  ```python
  sgd = SGDRegressor(max_iter=1000, tol=1e-3, early_stopping=True)
  ```
- **HistGradientBoostingRegressor:** Optimized for large datasets
- **LinearSVR:** Linear kernel, liblinear implementation
- **Ridge/Lasso (with solver='saga'):** Supports large-scale optimization
- **RandomForestRegressor with n_jobs=-1:** Parallelizable

**Avoid:**
- **SVR with RBF kernel:** O(n²) memory for kernel matrix, O(n³) training time
- **KNN:** O(n) prediction time per sample (exhaustive search)
- **Standard GradientBoostingRegressor:** Slower than Hist version

**Optimization strategies:**
```python
# 1. Reduce precision
X = X.astype(np.float32)

# 2. Use sparse matrices if applicable
from scipy.sparse import csr_matrix

# 3. Mini-batch training
sgd = SGDRegressor()
for batch in generate_batches(X, y, batch_size=10000):
    sgd.partial_fit(X_batch, y_batch)

# 4. Feature selection before training
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=20)
X_selected = selector.fit_transform(X, y)
```

**Q8: How do you detect and handle heteroscedasticity (non-constant variance) in residuals?**

**A:**
**Detection:**
1. **Visual:** Residual plot shows fan/cone shape
2. **Statistical tests:**
   - Breusch-Pagan test
   - White's test
   ```python
   from statsmodels.stats.diagnostic import het_breuschpagan
   _, p_value, _, _ = het_breuschpagan(residuals, X_train)
   if p_value < 0.05:
       print("Heteroscedasticity detected")
   ```

**Solutions:**
1. **Transform target variable:**
   ```python
   # Log transformation for right-skewed targets
   y_log = np.log1p(y)
   model.fit(X, y_log)
   predictions = np.expm1(model.predict(X_test))
   
   # Box-Cox transformation
   from scipy.stats import boxcox
   y_transformed, lambda_param = boxcox(y + 1)
   ```

2. **Weighted Least Squares:**
   ```python
   # Give less weight to high-variance observations
   from sklearn.linear_model import Ridge
   
   # Estimate variance at each point
   residuals_abs = np.abs(y_train - initial_model.predict(X_train))
   weights = 1 / (residuals_abs + 1e-8)
   
   # Fit with sample weights
   model.fit(X_train, y_train, sample_weight=weights)
   ```

3. **Robust regression:**
   ```python
   from sklearn.linear_model import HuberRegressor
   huber = HuberRegressor()
   ```

4. **Use models robust to heteroscedasticity:** Tree-based models naturally handle varying variance

**Q9: Compare time and space complexity for prediction in different regression models.**

**A:**
| Model | Training Time | Training Space | Prediction Time (per sample) | Prediction Space |
|-------|---------------|----------------|------------------------------|------------------|
| Linear/Ridge/Lasso | O(n·p²) | O(n·p) | O(p) | O(p) |
| SVR (RBF) | O(n²·p) to O(n³·p) | O(n²) | O(n_sv·p) | O(n_sv·p) |
| Decision Tree | O(n·p·log n) | O(n) | O(log n) | O(nodes) |
| Random Forest | O(n·p·log n·k) | O(k·n) | O(k·log n) | O(k·nodes) |
| Gradient Boosting | O(n·p·d·k) | O(n) | O(k·d) | O(k·nodes) |
| KNN | O(1) | O(n·p) | O(n·p) | O(n·p) |

Where: n=samples, p=features, k=trees/estimators, d=tree depth, n_sv=support vectors

**Production implications:**
- **Low latency needed:** Linear models or shallow trees
- **Large-scale prediction:** Avoid KNN, use linear models
- **Memory constrained:** Avoid RF with many trees, use linear models
- **High-dimensional:** Linear models (sparse solutions with Lasso)

**Q10: How would you diagnose why your regression model's R² is negative on the test set?**

**A:** **R² < 0 means the model performs worse than predicting the mean.** This indicates serious problems:

**Likely causes:**
1. **Severe overfitting:**
   - Training R² is high (>0.8), test R² negative
   - Solution: Add regularization, reduce complexity, collect more data

2. **Data leakage:**
   - Features contain information not available at prediction time
   - Check for target-derived features, future information
   - Solution: Recreate features without leakage

3. **Train-test distribution shift:**
   - Test data comes from different distribution than training
   - Example: Model trained on summer data, tested on winter
   - Solution: Ensure train/test stratification, check temporal effects

4. **Wrong features:**
   - Features have no predictive power for target
   - Solution: Feature selection, domain expertise, correlation analysis

5. **Prediction range issues (trees):**
   - Decision trees can't extrapolate beyond training range
   - If test target range is [100, 200] but training was [0, 50], predictions capped at 50
   - Solution: Ensure training covers expected test range

6. **Inappropriate feature scaling:**
   - Forgot to scale test data or used different scaler
   ```python
   # Wrong
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.fit_transform(X_test)  # ❌ Different scaling
   
   # Correct
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)  # ✓ Same scaling
   ```

**Debugging steps:**
```python
# 1. Check predictions vs actual
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

# 2. Compare distributions
print(f"Train target: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
print(f"Test target: mean={y_test.mean():.2f}, std={y_test.std():.2f}")

# 3. Check feature distributions
for i in range(X.shape[1]):
    print(f"Feature {i} - Train: {X_train[:, i].mean():.2f}, Test: {X_test[:, i].mean():.2f}")

# 4. Baseline comparison
baseline_pred = np.full_like(y_test, y_train.mean())
baseline_r2 = r2_score(y_test, baseline_pred)
print(f"Baseline R²: {baseline_r2:.3f}")  # Should be 0.0
```

---

## 3. Clustering

### Definition and Fundamentals

Clustering is an unsupervised learning technique that groups similar data points into clusters without labeled training data. The goal is to maximize intra-cluster similarity while minimizing inter-cluster similarity. Scikit-learn provides diverse clustering algorithms suited for different data structures: K-Means for spherical clusters, DBSCAN for arbitrary shapes with noise, hierarchical clustering for taxonomy discovery, and Gaussian Mixture Models for probabilistic soft assignments. Clustering is fundamental for customer segmentation, anomaly detection, data exploration, and dimensionality reduction preprocessing.

### 3.1 K-Means Clustering

K-Means partitions data into K clusters by iteratively assigning points to nearest centroids and updating centroids as cluster means. It assumes spherical clusters of similar size and is sensitive to initialization and outliers.

**Algorithm:**
1. Initialize K centroids randomly
2. Assign each point to nearest centroid (Euclidean distance)
3. Recompute centroids as mean of assigned points
4. Repeat steps 2-3 until convergence

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic clustering dataset
X, y_true = make_blobs(n_samples=1000, n_features=2, centers=4, 
                       cluster_std=1.5, random_state=42)

# Feature scaling is crucial for distance-based clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',       # Smart initialization (default, recommended)
    n_init=10,              # Number of times algorithm runs with different seeds
    max_iter=300,
    tol=1e-4,
    random_state=42,
    algorithm='lloyd'       # 'lloyd' or 'elkan' (faster for well-separated clusters)
)

# Fit and predict
labels = kmeans.fit_predict(X_scaled)

# Cluster properties
print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
print(f"Number of iterations: {kmeans.n_iter_}")

# Evaluate clustering quality
silhouette = silhouette_score(X_scaled, labels)
davies_bouldin = davies_bouldin_score(X_scaled, labels)
calinski_harabasz = calinski_harabasz_score(X_scaled, labels)

print(f"\nClustering Metrics:")
print(f"Silhouette Score: {silhouette:.3f}")  # Higher is better [-1, 1]
print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")  # Lower is better
print(f"Calinski-Harabasz Index: {calinski_harabasz:.3f}")  # Higher is better

# Cluster sizes
unique, counts = np.unique(labels, return_counts=True)
print(f"\nCluster sizes: {dict(zip(unique, counts))}")
```

**Finding Optimal K (Elbow Method + Silhouette Analysis):**

```python
# Elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans_temp.labels_))

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.grid(True)

ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
ax2.grid(True)

plt.tight_layout()
plt.savefig('optimal_k_analysis.png')

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal K based on Silhouette Score: {optimal_k}")
```

**Mini-Batch K-Means (for large datasets):**

```python
from sklearn.cluster import MiniBatchKMeans

# Much faster for large datasets (>10K samples)
mb_kmeans = MiniBatchKMeans(
    n_clusters=4,
    batch_size=100,         # Size of mini-batches
    max_iter=100,
    random_state=42
)

labels_mb = mb_kmeans.fit_predict(X_scaled)

print(f"Mini-Batch K-Means Inertia: {mb_kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels_mb):.3f}")
```

### 3.2 DBSCAN (Density-Based Spatial Clustering)

DBSCAN discovers clusters of arbitrary shape by grouping points in high-density regions, automatically identifying outliers as noise. Unlike K-Means, it doesn't require specifying the number of clusters and is robust to outliers.

**Key concepts:**
- **Core point:** Has at least `min_samples` neighbors within `eps` radius
- **Border point:** Within `eps` of a core point but has fewer than `min_samples` neighbors
- **Noise point:** Neither core nor border (labeled as -1)

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# DBSCAN clustering
dbscan = DBSCAN(
    eps=0.5,                # Maximum distance between two samples
    min_samples=5,          # Minimum points to form dense region
    metric='euclidean',     # Distance metric
    n_jobs=-1
)

labels_dbscan = dbscan.fit_predict(X_scaled)

# Cluster analysis
n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Core samples: {len(dbscan.core_sample_indices_)}")

# Evaluate (excluding noise points)
if n_clusters > 1:
    mask = labels_dbscan != -1
    silhouette = silhouette_score(X_scaled[mask], labels_dbscan[mask])
    print(f"Silhouette Score (excluding noise): {silhouette:.3f}")
```

**Finding Optimal eps (K-distance plot):**

```python
# Calculate distance to k-th nearest neighbor for each point
k = 5  # Should match min_samples
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)

# Sort distances to k-th neighbor
k_distances = np.sort(distances[:, k-1])

# Plot k-distance graph
plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{k}-th Nearest Neighbor Distance')
plt.title('K-distance Graph for eps Selection')
plt.grid(True)

# Look for "elbow" in curve - that's your optimal eps
# Typically where curve has sharpest increase
```

### 3.3 Hierarchical Clustering

Hierarchical clustering builds a tree of clusters (dendrogram) through agglomerative (bottom-up) or divisive (top-down) approaches. Agglomerative is more common and available in sklearn.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Agglomerative Clustering
agg_clust = AgglomerativeClustering(
    n_clusters=4,
    linkage='ward',         # 'ward', 'complete', 'average', 'single'
    metric='euclidean'      # Only for linkage='complete', 'average', 'single'
)

labels_agg = agg_clust.fit_predict(X_scaled)

print(f"Agglomerative Clustering Silhouette: {silhouette_score(X_scaled, labels_agg):.3f}")

# Create dendrogram
plt.figure(figsize=(12, 6))
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.savefig('dendrogram.png')
```

### 3.4 Gaussian Mixture Models (GMM)

GMM assumes data is generated from a mixture of Gaussian distributions, providing probabilistic cluster assignments (soft clustering). Unlike K-Means, it can model elliptical clusters and provides uncertainty estimates.

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=4,
    covariance_type='full',  # 'full', 'tied', 'diag', 'spherical'
    max_iter=100,
    n_init=10,
    random_state=42
)

gmm.fit(X_scaled)

# Hard assignment (like K-Means)
labels_gmm = gmm.predict(X_scaled)

# Soft assignment (probabilities)
probs = gmm.predict_proba(X_scaled)
print(f"Probability shape: {probs.shape}")  # (n_samples, n_components)

# Model quality
print(f"BIC (lower is better): {gmm.bic(X_scaled):.2f}")
print(f"AIC (lower is better): {gmm.aic(X_scaled):.2f}")
print(f"Log-likelihood: {gmm.score(X_scaled):.2f}")

# Finding optimal number of components
bic_scores = []
aic_scores = []
n_components_range = range(2, 11)

for n in n_components_range:
    gmm_temp = GaussianMixture(n_components=n, random_state=42)
    gmm_temp.fit(X_scaled)
    bic_scores.append(gmm_temp.bic(X_scaled))
    aic_scores.append(gmm_temp.aic(X_scaled))

optimal_n = n_components_range[np.argmin(bic_scores)]
print(f"Optimal components based on BIC: {optimal_n}")
```

### Common Pitfalls and Real-World Tips

**1. Feature Scaling is Critical:**
```python
# K-Means uses Euclidean distance - unscaled features dominate
# Wrong approach (no scaling)
kmeans_unscaled = KMeans(n_clusters=3)
kmeans_unscaled.fit(X)  # Feature with range [0, 1000] dominates [0, 1]

# Correct approach
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans_scaled = KMeans(n_clusters=3)
kmeans_scaled.fit(X_scaled)
```

**2. K-Means Assumptions:**
- Clusters are spherical and similar size
- Not suitable for elongated or irregular shapes
- Use DBSCAN or GMM for complex cluster shapes

**3. Handling Categorical Features:**
```python
# K-Means requires numerical features
# Use one-hot encoding or K-Modes for categorical data
from sklearn.preprocessing import OneHotEncoder

# For mixed data (numerical + categorical)
# Use Gower distance with custom implementations
```

**4. Curse of Dimensionality:**
```python
# High dimensions make distances less meaningful
# Apply PCA before clustering for high-dimensional data
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Retain 95% variance
X_reduced = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(X_reduced)
```

**5. Evaluating Clustering Without Ground Truth:**
```python
# Internal metrics (no true labels needed)
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Silhouette: -1 (worst) to 1 (best)
# Measures how similar points are to their cluster vs other clusters
silhouette = silhouette_score(X_scaled, labels)

# Davies-Bouldin: Lower is better
# Ratio of within-cluster to between-cluster distances
db_index = davies_bouldin_score(X_scaled, labels)

# External metrics (when true labels available)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ari = adjusted_rand_score(y_true, labels)  # Range [-1, 1], 0=random
nmi = normalized_mutual_info_score(y_true, labels)  # Range [0, 1]
```

**6. Dealing with Outliers:**
```python
# K-Means is sensitive to outliers
# Option 1: Use DBSCAN (marks outliers as noise)
# Option 2: Remove outliers before K-Means
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(X_scaled)
X_clean = X_scaled[outlier_labels == 1]

# Option 3: Use robust clustering (DBSCAN, GMM with covariance constraints)
```

**7. Computational Efficiency:**
```python
# For large datasets (>10K samples)
# Use MiniBatchKMeans (10-100x faster)
mb_kmeans = MiniBatchKMeans(n_clusters=5, batch_size=1000)

# For high-dimensional data
# Reduce dimensions first with PCA or feature selection
```

### Interview Questions and Answers

**Q1: Explain the difference between K-Means and DBSCAN. When would you use each?**

**A:**

**K-Means:**
- Assumes spherical clusters of similar size
- Requires pre-specifying K (number of clusters)
- Every point assigned to a cluster (no noise concept)
- Fast: O(n·k·i) where i=iterations
- Sensitive to outliers and initialization

**DBSCAN:**
- Discovers arbitrary-shaped clusters
- Automatically determines number of clusters
- Identifies outliers as noise points
- Requires tuning eps and min_samples
- Slower for high dimensions

**Use K-Means when:**
- Clusters are roughly spherical and similar size
- You know approximate number of clusters
- Speed is critical (large datasets)
- Example: Customer segmentation with balanced groups

**Use DBSCAN when:**
- Cluster shapes are irregular (e.g., moons, spirals)
- You don't know number of clusters
- Data contains significant noise/outliers
- Example: Geospatial clustering, anomaly detection

**Q2: How do you determine the optimal number of clusters in K-Means?**

**A:** Multiple approaches, use in combination:

**1. Elbow Method:**
- Plot K vs inertia (within-cluster sum of squares)
- Look for "elbow" where inertia decrease slows
- Limitation: Elbow can be ambiguous

**2. Silhouette Analysis:**
- Calculate silhouette score for different K values
- Choose K with highest silhouette score
- Range: [-1, 1], higher is better
- More reliable than elbow method

**3. Domain Knowledge:**
- Use business context (e.g., customer segments = product lines)
- Most reliable when available

**4. BIC/AIC (for GMM):**
- Lower is better
- Penalizes model complexity

**Code example:**
```python
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

optimal_k = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2
```

**Production tip:** Start with business requirements, validate with metrics.

**Q3: Your K-Means clustering produces one very large cluster and several tiny ones. What's wrong and how do you fix it?**

**A:** This indicates **imbalanced cluster sizes**, common causes:

**Causes:**
1. **Unscaled features:** One feature dominates distance calculations
2. **Outliers:** Pull centroids away from true cluster centers
3. **Wrong K:** Too many clusters specified
4. **Non-spherical data:** K-Means assumes spherical clusters

**Solutions:**

**1. Feature scaling (most common fix):**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**2. Remove outliers:**
```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05)
mask = iso.fit_predict(X) == 1
X_clean = X[mask]
```

**3. Try different K values:**
- Use elbow/silhouette analysis
- May be over-clustering

**4. Switch algorithm:**
- Use DBSCAN for arbitrary shapes
- Use GMM for elliptical clusters
- Use Agglomerative with different linkage

**5. Check data distribution:**
```python
# Visualize with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
```

**Q4: How would you implement customer segmentation for an e-commerce platform?**

**A:** Structured approach:

**1. Feature Engineering:**
```python
# RFM features (Recency, Frequency, Monetary)
features = pd.DataFrame({
    'days_since_last_purchase': ...,
    'num_purchases': ...,
    'total_spent': ...,
    'avg_order_value': ...,
    'num_product_categories': ...,
    'cart_abandonment_rate': ...,
    'days_as_customer': ...
})
```

**2. Preprocessing:**
```python
# Handle skewness
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
features_transformed = pt.fit_transform(features)

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_transformed)
```

**3. Determine optimal K:**
```python
# Combine silhouette + domain knowledge
# E.g., business might want 4-6 segments
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    silhouette_scores.append(silhouette_score(features_scaled, labels))
```

**4. Apply clustering:**
```python
optimal_k = 5  # Based on analysis + business input
kmeans = KMeans(n_clusters=optimal_k, n_init=20, random_state=42)
segments = kmeans.fit_predict(features_scaled)
```

**5. Profile segments:**
```python
# Inverse transform centroids to original scale
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids_original, columns=features.columns)

# Name segments based on characteristics
# E.g., "High-Value Regulars", "Bargain Hunters", "At-Risk", etc.

# Analyze segment sizes and characteristics
for seg in range(optimal_k):
    mask = segments == seg
    print(f"\nSegment {seg}: {mask.sum()} customers")
    print(features[mask].describe())
```

**6. Validation:**
- Check if segments are actionable (clear differences)
- Stability test: Re-run with different random_state
- Business review: Do segments make sense?

**Q5: What's the difference between hard and soft clustering? Give examples of each.**

**A:**

**Hard Clustering:**
- Each point assigned to exactly one cluster
- Definitive assignments
- **Examples:** K-Means, DBSCAN, Agglomerative

```python
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)  # Each point gets one label
# Point belongs 100% to one cluster
```

**Soft Clustering:**
- Each point has probability distribution over clusters
- Captures uncertainty
- **Example:** Gaussian Mixture Models

```python
gmm = GaussianMixture(n_components=3)
gmm.fit(X)

# Hard assignment (like K-Means)
labels = gmm.predict(X)

# Soft assignment (probabilities)
probs = gmm.predict_proba(X)
# Example: Point belongs 70% to cluster 0, 25% to cluster 1, 5% to cluster 2
```

**When to use soft clustering:**
- Overlapping clusters (e.g., customer behaviors)
- Need confidence scores for downstream tasks
- Probabilistic modeling requirements
- Example: Document topic modeling, ambiguous customer segments

**When to use hard clustering:**
- Clear separation between clusters
- Simpler interpretation needed
- Memory constraints (soft clustering stores probabilities)

**Q6: How do you handle high-dimensional data in clustering?**

**A:** High dimensions cause **distance metrics to become less meaningful** (curse of dimensionality). Solutions:

**1. Dimensionality Reduction:**
```python
from sklearn.decomposition import PCA

# Reduce to components explaining 95% variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)

# Then apply clustering
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(X_reduced)
```

**2. Feature Selection:**
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top k features
# (requires pseudo-labels from initial clustering or domain knowledge)
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, initial_labels)
```

**3. Use algorithms robust to high dimensions:**
- **Spectral Clustering:** Works well with high dimensions
- **DBSCAN with appropriate metric:** Cosine distance for sparse data

```python
from sklearn.cluster import DBSCAN

# Use cosine distance for text/sparse data
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
labels = dbscan.fit_predict(X_sparse)
```

**4. Subspace clustering:**
- Cluster in different feature subspaces
- Use ensemble methods

**5. Autoencoders (deep learning):**
- Learn compressed representation
- Cluster in latent space

**Practical guideline:**
- **< 20 features:** Direct clustering usually fine
- **20-100 features:** Consider PCA
- **> 100 features:** PCA/feature selection essential

**Q7: You applied DBSCAN and got 90% of points labeled as noise. What went wrong?**

**A:** Problem: **eps is too small** or **min_samples is too large**.

**Diagnosis and fixes:**

**1. eps too small:**
```python
# Points can't find enough neighbors within eps radius

# Solution: Use k-distance plot to find appropriate eps
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)

# Plot k-th nearest neighbor distances
k_distances = np.sort(distances[:, 4])  # 5th neighbor (index 4)
plt.plot(k_distances)
plt.ylabel('5-th Nearest Neighbor Distance')

# Choose eps where curve has sharp increase (elbow)
# Typically between 25th-75th percentile
optimal_eps = np.percentile(k_distances, 75)
```

**2. min_samples too large:**
```python
# Too strict density requirement

# Rule of thumb: min_samples = 2 × dimensions (for low-D)
# For most cases: min_samples = 5-10

# Try smaller value
dbscan_fixed = DBSCAN(eps=optimal_eps, min_samples=5)
```

**3. Data not scaled:**
```python
# Features with different scales distort distances
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
```

**4. Data is uniformly distributed:**
- No natural clusters exist
- DBSCAN correctly identifies lack of density patterns
- Consider: Is clustering appropriate for this data?

**Debugging steps:**
```python
# Check distance distribution
from sklearn.metrics.pairwise import euclidean_distances

distances_matrix = euclidean_distances(X_scaled)
print(f"Mean distance: {distances_matrix.mean():.3f}")
print(f"Median distance: {np.median(distances_matrix):.3f}")

# Start with eps around median distance
```

**Q8: How would you evaluate clustering quality when you don't have ground truth labels?**

**A:** Use **internal validation metrics** that assess cluster quality from data structure:

**1. Silhouette Score (most common):**
```python
from sklearn.metrics import silhouette_score, silhouette_samples

# Overall score
silhouette_avg = silhouette_score(X_scaled, labels)
# Range: [-1, 1], aim for > 0.5

# Per-sample scores (identify poorly clustered points)
sample_scores = silhouette_samples(X_scaled, labels)

# Analyze per-cluster
for cluster in np.unique(labels):
    cluster_scores = sample_scores[labels == cluster]
    print(f"Cluster {cluster}: {cluster_scores.mean():.3f}")
```

**Interpretation:**
- **> 0.7:** Strong structure
- **0.5-0.7:** Reasonable structure
- **0.25-0.5:** Weak structure
- **< 0.25:** No substantial structure

**2. Davies-Bouldin Index:**
```python
from sklearn.metrics import davies_bouldin_score

# Lower is better (minimum = 0)
db_score = davies_bouldin_score(X_scaled, labels)
```

**3. Calinski-Harabasz Index:**
```python
from sklearn.metrics import calinski_harabasz_score

# Higher is better (ratio of between/within cluster dispersion)
ch_score = calinski_harabasz_score(X_scaled, labels)
```

**4. Visual inspection:**
```python
from sklearn.decomposition import PCA

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
plt.title('Cluster Visualization')
```

**5. Stability analysis:**
```python
# Run clustering multiple times with different random seeds
# Good clusters should be stable

scores = []
for seed in range(10):
    kmeans = KMeans(n_clusters=5, random_state=seed)
    labels_temp = kmeans.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, labels_temp))

print(f"Silhouette stability: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
# Low std indicates stable clusters
```

**6. Business validation:**
- Most important for production
- Do clusters make business sense?
- Are they actionable?
- Can domain experts interpret them?

**Practical approach:** Combine multiple metrics + visual inspection + domain validation

**Q9: K-Means vs Gaussian Mixture Models - when would GMM be preferred?**

**A:**

**Choose GMM over K-Means when:**

**1. Elliptical/Non-spherical clusters:**
```python
# K-Means assumes spherical clusters
# GMM can model elliptical shapes via covariance matrices

gmm = GaussianMixture(n_components=3, covariance_type='full')
# 'full': Each cluster has its own general covariance matrix
# 'tied': All clusters share same covariance
# 'diag': Diagonal covariance (axis-aligned ellipses)
# 'spherical': Same as K-Means
```

**2. Soft clustering needed:**
```python
# Get probability distributions instead of hard assignments
probs = gmm.predict_proba(X)

# Use cases:
# - Uncertain classifications (e.g., ambiguous customer behaviors)
# - Weighted recommendations based on cluster probabilities
# - Anomaly detection (low probability across all clusters)
```

**3. Clusters have different sizes/densities:**
- K-Means assumes similar cluster variances
- GMM adapts to different cluster spreads

**4. Model selection with BIC/AIC:**
```python
# GMM provides statistical model selection
bic_scores = [GaussianMixture(n_components=k).fit(X).bic(X) 
              for k in range(2, 11)]
optimal_k = np.argmin(bic_scores) + 2
```

**5. Generative modeling:**
- GMM can generate new samples
- K-Means is discriminative only

**Trade-offs:**
- **GMM:** Slower, more parameters to tune, can overfit
- **K-Means:** Faster, simpler, more robust for simple clusters

**Practical rule:** Try K-Means first (simpler, faster). If results poor, try GMM.

**Q10: How would you detect and handle outliers before clustering?**

**A:** Outliers can severely distort clustering, especially K-Means. Strategies:

**1. Isolation Forest (recommended for high dimensions):**
```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    contamination=0.1,      # Expected proportion of outliers
    random_state=42
)

outlier_labels = iso_forest.fit_predict(X_scaled)
# 1 = inlier, -1 = outlier

X_clean = X_scaled[outlier_labels == 1]
print(f"Removed {(outlier_labels == -1).sum()} outliers")
```

**2. Statistical methods:**
```python
# Z-score method (for normally distributed features)
from scipy import stats

z_scores = np.abs(stats.zscore(X_scaled))
threshold = 3  # Points beyond 3 std devs

outlier_mask = (z_scores < threshold).all(axis=1)
X_clean = X_scaled[outlier_mask]

# IQR method (robust to non-normal distributions)
Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)
IQR = Q3 - Q1

outlier_mask = ((X >= Q1 - 1.5*IQR) & (X <= Q3 + 1.5*IQR)).all(axis=1)
X_clean = X[outlier_mask]
```

**3. DBSCAN for outlier identification:**
```python
# Use DBSCAN first to identify noise, then K-Means on inliers
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

inliers = X_scaled[labels_dbscan != -1]
kmeans = KMeans(n_clusters=3)
labels_final = kmeans.fit_predict(inliers)
```

**4. Local Outlier Factor:**
```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outlier_labels = lof.fit_predict(X_scaled)

X_clean = X_scaled[outlier_labels == 1]
```

**5. Robust clustering algorithms:**
```python
# Use algorithms inherently robust to outliers
# - DBSCAN: Marks outliers as noise
# - GMM with regularization
# - Spectral clustering
```

**Practical workflow:**
```python
# 1. Visualize to understand outlier nature
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
plt.scatter(X_2d[:, 0], X_2d[:, 1])

# 2. Apply outlier detection
iso_forest = IsolationForest(contamination=0.05)
outlier_labels = iso_forest.fit_predict(X_scaled)

# 3. Analyze outliers before removing
outliers = X_scaled[outlier_labels == -1]
print(f"Outlier characteristics:\n{pd.DataFrame(outliers).describe()}")

# 4. Decide: remove, transform, or use robust algorithm
# - Remove if clearly noise/errors
# - Keep if legitimate rare cases (use DBSCAN)
# - Transform if due to skewness (log, Box-Cox)
```

**Important:** Don't blindly remove outliers. They might represent:
- Rare but important patterns (fraud, VIP customers)
- Data quality issues (need fixing at source)
- Natural variation (should keep)

---

## 4. Dimensionality Reduction

### Definition and Fundamentals

Dimensionality reduction transforms high-dimensional data into lower dimensions while preserving important structure. This addresses the curse of dimensionality, reduces computational cost, enables visualization, removes noise, and mitigates multicollinearity. Scikit-learn provides linear methods (PCA, TruncatedSVD) that preserve global structure through linear transformations, and manifold learning techniques (t-SNE, UMAP-style alternatives) that preserve local neighborhood relationships for visualization.

### 4.1 Principal Component Analysis (PCA)

PCA finds orthogonal directions (principal components) of maximum variance in data. It's a linear transformation that projects data onto a lower-dimensional space, with components ordered by explained variance.

**Key Concepts:**
- **Eigenvectors:** Directions of maximum variance (principal components)
- **Eigenvalues:** Amount of variance explained by each component
- **Variance preservation:** First k components capture most dataset variance

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

# Load high-dimensional dataset (64 features)
digits = load_digits()
X, y = digits.data, digits.target

# PCA requires scaled features (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize PCA
pca = PCA(
    n_components=0.95,      # Keep components explaining 95% variance
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_pca.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
print(f"Number of components: {pca.n_components_}")

# Component properties
print(f"\nComponent shapes:")
print(f"Components (loadings): {pca.components_.shape}")  # (n_components, n_features)
print(f"Explained variance: {pca.explained_variance_[:5]}")  # First 5 components
print(f"Mean: {pca.mean_.shape}")

# Inverse transform (reconstruct original data)
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"\nReconstruction error (MSE): {reconstruction_error:.4f}")
```

**Scree Plot and Cumulative Variance:**

```python
# Fit PCA with all components
pca_full = PCA()
pca_full.fit(X_scaled)

# Plot explained variance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
ax1.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         pca_full.explained_variance_ratio_, 'bo-')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Scree Plot')
ax1.grid(True)

# Cumulative variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
ax2.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('Cumulative Variance Explained')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# Find number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")
```

**Visualization in 2D/3D:**

```python
# Reduce to 2D for visualization
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA: First Two Principal Components')
plt.colorbar(scatter, label='Digit Class')
plt.savefig('pca_2d_visualization.png')

# 3D visualization
from mpl_toolkits.mplot3d import Axes3D

pca_3d = PCA(n_components=3, random_state=42)
X_3d = pca_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap='tab10')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title(f'PCA 3D (Total variance: {pca_3d.explained_variance_ratio_.sum():.2%})')
plt.colorbar(scatter)
```

**Feature Contribution Analysis:**

```python
# Which original features contribute most to each PC?
import pandas as pd

n_top_features = 5
feature_names = [f'pixel_{i}' for i in range(X.shape[1])]

for i in range(3):  # First 3 components
    # Get absolute loadings for component i
    loadings = np.abs(pca.components_[i])
    top_indices = np.argsort(loadings)[-n_top_features:][::-1]
    
    print(f"\nPC{i+1} - Top {n_top_features} contributing features:")
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {pca.components_[i, idx]:.3f}")
```

**PCA for Preprocessing:**

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: Scale -> PCA -> Classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

pipeline.fit(X_train, y_train)

print(f"Pipeline accuracy: {pipeline.score(X_test, y_test):.3f}")
print(f"Components used: {pipeline.named_steps['pca'].n_components_}")

# Compare with no PCA
pipeline_no_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

pipeline_no_pca.fit(X_train, y_train)
print(f"No PCA accuracy: {pipeline_no_pca.score(X_test, y_test):.3f}")
```

### 4.2 Incremental PCA

Incremental PCA processes data in mini-batches, enabling dimensionality reduction for datasets too large to fit in memory.

```python
from sklearn.decomposition import IncrementalPCA

# Useful when X doesn't fit in memory
n_components = 20
batch_size = 200

ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

# Fit in batches
for i in range(0, X_scaled.shape[0], batch_size):
    batch = X_scaled[i:i+batch_size]
    ipca.partial_fit(batch)

X_ipca = ipca.transform(X_scaled)

print(f"Incremental PCA - Components: {X_ipca.shape[1]}")
print(f"Explained variance: {ipca.explained_variance_ratio_.sum():.3f}")

# Compare with standard PCA
pca_standard = PCA(n_components=n_components)
X_pca_standard = pca_standard.fit_transform(X_scaled)

print(f"\nVariance difference: {abs(ipca.explained_variance_ratio_.sum() - pca_standard.explained_variance_ratio_.sum()):.4f}")
```

### 4.3 Kernel PCA

Kernel PCA applies the kernel trick to perform non-linear dimensionality reduction, capturing complex non-linear relationships.

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles

# Generate non-linear dataset
X_circles, y_circles = make_circles(n_samples=1000, factor=0.3, noise=0.05, random_state=42)

# Linear PCA (won't separate circles)
pca_linear = PCA(n_components=2)
X_pca_linear = pca_linear.fit_transform(X_circles)

# Kernel PCA with RBF kernel
kpca = KernelPCA(
    n_components=2,
    kernel='rbf',           # 'linear', 'poly', 'rbf', 'sigmoid', 'cosine'
    gamma=10,               # Kernel coefficient
    fit_inverse_transform=True,  # Enable inverse transform (approximate)
    random_state=42
)

X_kpca = kpca.fit_transform(X_circles)

# Visualize comparison
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

ax1.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis')
ax1.set_title('Original Data')

ax2.scatter(X_pca_linear[:, 0], X_pca_linear[:, 1], c=y_circles, cmap='viridis')
ax2.set_title('Linear PCA')

ax3.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y_circles, cmap='viridis')
ax3.set_title('Kernel PCA (RBF)')

plt.tight_layout()
```

### 4.4 Truncated SVD (LSA for text)

Truncated SVD (Singular Value Decomposition) works with sparse matrices and doesn't require centering, making it ideal for text data and recommender systems.

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# Load text data
newsgroups = fetch_20newsgroups(subset='train', categories=['sci.space', 'rec.sport.baseball'],
                                remove=('headers', 'footers', 'quotes'))

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(newsgroups.data)

print(f"Original sparse matrix shape: {X_tfidf.shape}")
print(f"Sparsity: {(1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])):.2%}")

# Truncated SVD (Latent Semantic Analysis)
svd = TruncatedSVD(
    n_components=100,       # Number of topics/concepts
    algorithm='randomized', # 'randomized' or 'arpack' (randomized is faster)
    n_iter=5,
    random_state=42
)

X_svd = svd.fit_transform(X_tfidf)

print(f"\nReduced dimensions: {X_svd.shape}")
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")

# Analyze topics (top words per component)
feature_names = vectorizer.get_feature_names_out()
n_top_words = 10

for i in range(3):  # First 3 components
    top_indices = np.argsort(svd.components_[i])[-n_top_words:][::-1]
    top_words = [feature_names[idx] for idx in top_indices]
    print(f"\nTopic {i+1}: {', '.join(top_words)}")
```

### 4.5 t-SNE (for visualization only)

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a non-linear technique that preserves local structure, excellent for visualization but not for feature reduction or downstream tasks.

```python
from sklearn.manifold import TSNE

# Important: Apply PCA first for high-dimensional data (speed + noise reduction)
pca_pre = PCA(n_components=50, random_state=42)
X_pca_pre = pca_pre.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(
    n_components=2,         # Typically 2 or 3 for visualization
    perplexity=30,          # Balance between local and global structure (5-50)
    learning_rate=200,      # Step size (10-1000)
    n_iter=1000,            # Number of iterations
    random_state=42,
    init='pca'              # Initialize with PCA (more stable)
)

X_tsne = tsne.fit_transform(X_pca_pre)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization')
plt.colorbar(scatter, label='Digit Class')
plt.savefig('tsne_visualization.png')

print(f"KL divergence (lower is better): {tsne.kl_divergence_:.3f}")
```

**CRITICAL: t-SNE limitations:**
- Don't use for preprocessing (only visualization)
- Different runs produce different results
- Perplexity significantly affects results
- Computationally expensive (O(n²))

### Common Pitfalls and Real-World Tips

**1. Feature Scaling for PCA:**
```python
# ALWAYS scale before PCA
# Wrong: Features with larger variance dominate PCs
pca_wrong = PCA(n_components=5)
pca_wrong.fit(X)  # ❌ No scaling

# Correct
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca_correct = PCA(n_components=5)
pca_correct.fit(X_scaled)  # ✓
```

**2. Choosing Number of Components:**
```python
# Strategy 1: Explain specific variance (e.g., 95%)
pca = PCA(n_components=0.95)

# Strategy 2: Fixed number based on task
# - Visualization: 2-3 components
# - Downstream ML: Cross-validate

# Strategy 3: Elbow method (where variance gain diminishes)
pca_full = PCA().fit(X_scaled)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
```

**3. PCA Before or After Train-Test Split:**
```python
# ALWAYS fit PCA on training data only
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Wrong: Data leakage
pca_wrong = PCA(n_components=10)
X_all_pca = pca_wrong.fit_transform(X_scaled)  # ❌ Uses test data info

# Correct
pca_correct = PCA(n_components=10)
X_train_pca = pca_correct.fit_transform(X_train)  # ✓ Fit on train only
X_test_pca = pca_correct.transform(X_test)        # ✓ Transform test
```

**4. Sparse Data (Text, Recommender Systems):**
```python
# Don't use PCA for sparse data (it densifies the matrix)
# Use TruncatedSVD instead

from scipy.sparse import csr_matrix

X_sparse = csr_matrix(X_tfidf)  # Keep sparse

# Wrong: Memory explosion
# pca = PCA(n_components=100)
# pca.fit(X_sparse.toarray())  # ❌ Creates dense array

# Correct
svd = TruncatedSVD(n_components=100)
svd.fit(X_sparse)  # ✓ Maintains sparsity
```

**5. PCA for Multicollinearity:**
```python
# PCA creates uncorrelated features (orthogonal components)
# Useful before regression with correlated features

from sklearn.linear_model import Ridge

# Original features are correlated
corr_matrix = np.corrcoef(X_scaled.T)
high_corr = np.sum(np.abs(corr_matrix) > 0.8) - X_scaled.shape[1]
print(f"Highly correlated feature pairs: {high_corr // 2}")

# Apply PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# PCA components are uncorrelated
corr_pca = np.corrcoef(X_pca.T)
print(f"Max off-diagonal correlation: {np.max(np.abs(corr_pca - np.eye(X_pca.shape[1]))):.6f}")
```

**6. Interpreting Principal Components:**
```python
# PCA components are linear combinations - hard to interpret
# Use loadings to understand which features contribute

def get_top_features_per_component(pca, feature_names, n_top=5):
    """Show top contributing features for each PC"""
    for i, component in enumerate(pca.components_):
        top_indices = np.argsort(np.abs(component))[-n_top:][::-1]
        print(f"\nPC{i+1}:")
        for idx in top_indices:
            print(f"  {feature_names[idx]}: {component[idx]:.3f}")

# Usage
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
get_top_features_per_component(pca, feature_names)
```

**7. When NOT to Use PCA:**
```python
# Don't use PCA when:
# 1. Features are already uncorrelated
# 2. Interpretability is critical (PCA creates abstract features)
# 3. Data is sparse (text) - use TruncatedSVD
# 4. Non-linear relationships - use Kernel PCA or autoencoders
# 5. Few features (<20) - overhead not worth it
```

### Interview Questions and Answers

**Q1: Explain the difference between PCA and t-SNE. When would you use each?**

**A:**

**PCA (Principal Component Analysis):**
- **Linear** transformation maximizing variance
- **Global structure:** Preserves large-scale relationships
- **Deterministic:** Same result every run
- **Fast:** O(n × p²) for n samples, p features
- **Use for:** Preprocessing, feature reduction, noise reduction
- **Output:** Can be used for downstream ML tasks

**t-SNE:**
- **Non-linear** manifold learning preserving local structure
- **Local structure:** Focuses on neighborhood relationships
- **Stochastic:** Different results each run
- **Slow:** O(n²) complexity
- **Use for:** Visualization only (2D/3D)
- **Output:** NOT suitable for downstream tasks

**Decision tree:**
```python
# High-dimensional data (100+ features)
# ↓
# Apply PCA (reduce to 20-50 components) for ML pipeline
# ↓
# Apply t-SNE (reduce to 2 components) for visualization only

# Pipeline example
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X_scaled)

# Train model on PCA output
model.fit(X_reduced, y)

# Separately: visualize with t-SNE
tsne = TSNE(n_components=2)
X_vis = tsne.fit_transform(X_reduced)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y)
```

**Key point:** PCA for dimensionality reduction, t-SNE for visualization.

**Q2: You applied PCA and the first component explains 90% of variance. Is this good or bad?**

**A:** **Context-dependent**, but often a **warning sign:**

**Potential issues:**
1. **Features not scaled:** One feature with large range dominates
   ```python
   # Always scale first
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **One feature is linear combination of others:** Multicollinearity
   - Solution: Remove redundant features before PCA

3. **Data has very low intrinsic dimensionality:** Might not need PCA at all

4. **Data is artificially inflated:** Check for duplicate or derived features

**When it's acceptable:**
- Image data with constant backgrounds (background dominates variance)
- Time series with strong trend component

**Validation:**
```python
# Check if single component actually captures meaningful info
pca = PCA(n_components=1)
X_1pc = pca.fit_transform(X_scaled)

# Reconstruct and visualize
X_reconstructed = pca.inverse_transform(X_1pc)

# If reconstruction looks like original, 1 PC is sufficient
# If reconstruction loses important details, investigate feature scaling
```

**Q3: How do you decide the optimal number of principal components?**

**A:** Multiple strategies, use in combination:

**1. Variance threshold (most common):**
```python
# Keep components explaining 95% variance (rule of thumb)
pca = PCA(n_components=0.95)
pca.fit(X_scaled)
print(f"Components selected: {pca.n_components_}")

# Or 90% for more aggressive reduction
# Or 99% for minimal information loss
```

**2. Elbow method:**
```python
pca_full = PCA().fit(X_scaled)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)

plt.plot(cumsum)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')

# Look for elbow where curve flattens
```

**3. Cross-validation with downstream task:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

scores = []
n_components_range = range(5, 51, 5)

for n in n_components_range:
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X_train_scaled)
    
    lr = LogisticRegression(max_iter=1000)
    score = cross_val_score(lr, X_pca, y_train, cv=5, scoring='accuracy').mean()
    scores.append(score)

optimal_n = n_components_range[np.argmax(scores)]
print(f"Optimal components: {optimal_n}")
```

**4. Kaiser criterion:**
- Keep components with eigenvalue > 1 (more variance than original feature)
- Less reliable for scaled data

**5. Computational constraints:**
- Visualization: 2-3 components
- Fast prediction: 10-20 components
- Maximum accuracy: 50-100 components

**Practical approach:**
1. Start with 95% variance threshold
2. Validate with cross-validation
3. Consider computational trade-offs

**Q4: Can you use PCA for categorical features? If not, what are alternatives?**

**A:** **No, PCA requires continuous numerical features** (works with covariance matrix).

**Alternatives for categorical data:**

**1. Encode first, then PCA:**
```python
from sklearn.preprocessing import OneHotEncoder

# One-hot encode categorical features
ohe = OneHotEncoder(sparse_output=False)
X_encoded = ohe.fit_transform(X_categorical)

# Then apply PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_encoded)
```

**2. Multiple Correspondence Analysis (MCA):**
```python
# Similar to PCA but for categorical data
# Not in sklearn, use prince library
# import prince
# mca = prince.MCA(n_components=5)
# mca.fit(X_categorical)
```

**3. Mixed data (numerical + categorical):**
```python
# Approach 1: Encode categorical, scale all, then PCA
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(), categorical_features)
])

X_processed = preprocessor.fit_transform(X)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_processed)

# Approach 2: Use feature hashing for high-cardinality categoricals
from sklearn.feature_extraction import FeatureHasher

# Approach 3: For tree-based models, skip PCA (handles mixed data natively)
from sklearn.ensemble import RandomForestClassifier
# Trees don't need PCA
```

**4. Autoencoders (deep learning):**
- Can handle any data type
- Non-linear reduction
- More complex but flexible

**Q5: Your PCA preprocessing improved training accuracy but hurt test accuracy. What happened?**

**A:** **Data leakage** - PCA was fit on entire dataset including test data.

**Problem:**
```python
# WRONG: Leakage
X_scaled = scaler.fit_transform(X)  # Uses test data statistics
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)  # Learns from test data

X_train, X_test, y_train, y_test = train_test_split(X_pca, y)
# Test data info leaked into training
```

**Correct approach:**
```python
# 1. Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Fit preprocessing on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ✓ Train only
X_test_scaled = scaler.transform(X_test)        # ✓ Use train statistics

# 3. Fit PCA on training data only
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)  # ✓ Train only
X_test_pca = pca.transform(X_test_scaled)        # ✓ Use train components

# 4. Train model
model.fit(X_train_pca, y_train)

# 5. Evaluate
print(f"Train: {model.score(X_train_pca, y_train):.3f}")
print(f"Test: {model.score(X_test_pca, y_test):.3f}")
```

**Use Pipeline to prevent mistakes:**
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', LogisticRegression())
])

# Pipeline handles splitting correctly
pipeline.fit(X_train, y_train)
print(f"Test accuracy: {pipeline.score(X_test, y_test):.3f}")
```

**Q6: When would you choose TruncatedSVD over PCA?**

**A:** **Use TruncatedSVD for sparse data** (text, recommender systems).

**Key differences:**

| Aspect | PCA | TruncatedSVD |
|--------|-----|--------------|
| **Centering** | Centers data (mean=0) | No centering |
| **Sparse data** | Densifies matrix (memory issue) | Maintains sparsity |
| **Use case** | Dense numerical features | Sparse features (text, ratings) |
| **Algorithm** | Eigendecomposition | SVD |

**Example:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA

# Text data → TF-IDF (sparse matrix)
vectorizer = TfidfVectorizer(max_features=10000)
X_tfidf = vectorizer.fit_transform(documents)

print(f"Matrix shape: {X_tfidf.shape}")
print(f"Sparsity: {(1 - X_tfidf.nnz/(X_tfidf.shape[0]*X_tfidf.shape[1])):.2%}")

# Wrong: PCA on sparse data
# pca = PCA(n_components=100)
# X_pca = pca.fit_transform(X_tfidf.toarray())  # ❌ Memory explosion!

# Correct: TruncatedSVD
svd = TruncatedSVD(n_components=100)
X_svd = svd.fit_transform(X_tfidf)  # ✓ Keeps sparse
```

**When sparse:**
- Text data (TF-IDF, CountVectorizer)
- User-item matrices (collaborative filtering)
- High-dimensional binary features

**When dense:**
- Image pixels
- Sensor readings
- Traditional tabular data

**Q7: How would you use PCA to detect outliers in high-dimensional data?**

**A:** Use **reconstruction error** - outliers have large reconstruction error because they don't fit the learned principal components.

**Method:**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply PCA (keep fewer components than features)
pca = PCA(n_components=0.95)  # Or fixed number
X_pca = pca.fit_transform(X_scaled)

# 3. Reconstruct data
X_reconstructed = pca.inverse_transform(X_pca)

# 4. Calculate reconstruction error per sample
reconstruction_errors = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

# 5. Identify outliers (threshold based on quantile or std)
threshold = np.percentile(reconstruction_errors, 95)  # Top 5% as outliers
outliers = reconstruction_errors > threshold

print(f"Number of outliers: {outliers.sum()}")
print(f"Outlier indices: {np.where(outliers)[0]}")

# 6. Visualize
plt.figure(figsize=(10, 6))
plt.hist(reconstruction_errors, bins=50)
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.title('PCA-based Outlier Detection')
```

**Why it works:**
- Normal points lie close to principal component space
- Outliers deviate from learned patterns → high reconstruction error

**Alternative: Mahalanobis distance in PCA space:**
```python
# Distance from center in PCA space
from scipy.spatial.distance import mahalanobis

mean_pca = X_pca.mean(axis=0)
cov_pca = np.cov(X_pca.T)
cov_inv = np.linalg.inv(cov_pca)

distances = [mahalanobis(x, mean_pca, cov_inv) for x in X_pca]
outliers_maha = np.array(distances) > np.percentile(distances, 95)
```

**Q8: Explain the trade-off between number of components and model performance.**

**A:** Classic **bias-variance trade-off** in dimensionality reduction:

**Too few components (high bias):**
- **Underfitting:** Lost important information
- Model can't capture data complexity
- Low train and test performance
- Fast computation
- Example: 5 components for 1000-dimensional data

**Too many components (high variance):**
- **Overfitting:** Included noise in model
- Model fits training data too closely
- High train, low test performance
- Slow computation
- Example: 500 components for 1000-dimensional data

**Optimal zone:**
- Captures signal, excludes noise
- Good generalization
- Example: 50-100 components explaining 95% variance

**Empirical analysis:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Test different numbers of components
n_components_range = [5, 10, 20, 30, 50, 70, 100]
train_scores = []
test_scores = []
cv_scores = []

for n in n_components_range:
    # Apply PCA
    pca = PCA(n_components=n, random_state=42)
    X_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train model
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_pca, y_train)
    
    # Evaluate
    train_scores.append(lr.score(X_pca, y_train))
    test_scores.append(lr.score(X_test_pca, y_test))
    
    # Cross-validation
    cv_score = cross_val_score(lr, X_pca, y_train, cv=5, scoring='accuracy').mean()
    cv_scores.append(cv_score)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, train_scores, 'o-', label='Train')
plt.plot(n_components_range, test_scores, 'o-', label='Test')
plt.plot(n_components_range, cv_scores, 'o-', label='CV')
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.title('Model Performance vs Number of Components')
plt.legend()
plt.grid(True)

optimal_n = n_components_range[np.argmax(cv_scores)]
print(f"Optimal number of components: {optimal_n}")
```

**Practical guidelines:**
- Start with 95% variance as baseline
- If overfitting: Reduce components (more regularization)
- If underfitting: Increase components (less regularization)
- Always validate with cross-validation, not just train/test

**Q9: How does PCA help with multicollinearity in linear regression?**

**A:** PCA creates **orthogonal (uncorrelated) features**, eliminating multicollinearity completely.

**Problem with multicollinearity:**
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Create correlated features
X_corr = np.random.randn(100, 3)
X_corr[:, 2] = X_corr[:, 0] + X_corr[:, 1] + np.random.randn(100) * 0.1

# Check correlation
print("Correlation matrix:")
print(np.corrcoef(X_corr.T))

# Regression with multicollinearity
lr = LinearRegression()
lr.fit(X_corr, y)

# Unstable coefficients, high variance
print(f"Coefficients: {lr.coef_}")
# Small data changes → large coefficient changes
```

**Solution with PCA:**
```python
# Apply PCA
pca = PCA(n_components=2)  # Reduce from 3 to 2
X_pca = pca.fit_transform(X_corr)

# Check correlation of PCs
print("\nPCA correlation matrix:")
print(np.corrcoef(X_pca.T))  # Perfect 0s off-diagonal

# Regression on PCs
lr_pca = LinearRegression()
lr_pca.fit(X_pca, y)

# Stable coefficients
print(f"PCA Coefficients: {lr_pca.coef_}")
```

**Benefits:**
- **Stable estimates:** No coefficient inflation
- **Better generalization:** Reduced variance
- **Interpretability loss:** PCs are linear combinations

**Alternative to PCA:**
```python
# If interpretability matters, use Ridge instead
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_corr, y)
# Ridge shrinks correlated coefficients without eliminating features
```

**When to use PCA vs Ridge:**
- **PCA:** Severe multicollinearity (correlations > 0.9), many features
- **Ridge:** Moderate multicollinearity, interpretability important

**Q10: You have 100K samples with 50K features (text data). How would you approach dimensionality reduction?**

**A:** **High-dimensional sparse data** requires careful strategy for memory and computation efficiency.

**Approach:**

**1. Use Truncated SVD (not PCA):**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# TF-IDF vectorization (sparse matrix)
vectorizer = TfidfVectorizer(
    max_features=50000,
    min_df=5,              # Ignore rare terms
    max_df=0.8,            # Ignore too common terms
    stop_words='english'
)

X_tfidf = vectorizer.fit_transform(documents)  # Sparse matrix

print(f"Sparse matrix: {X_tfidf.shape}")
print(f"Memory: {X_tfidf.data.nbytes / 1e6:.1f} MB")  # Much less than dense

# Apply TruncatedSVD (maintains sparsity)
svd = TruncatedSVD(
    n_components=300,      # Start with 100-500 for text
    algorithm='randomized', # Faster for large matrices
    n_iter=5,
    random_state=42
)

X_reduced = svd.fit_transform(X_tfidf)

print(f"Reduced: {X_reduced.shape}")
print(f"Explained variance: {svd.explained_variance_ratio_.sum():.3f}")
```

**2. Feature selection before reduction:**
```python
from sklearn.feature_selection import SelectKBest, chi2

# Remove low-information features first
selector = SelectKBest(chi2, k=10000)  # Keep top 10K features
X_selected = selector.fit_transform(X_tfidf, y)

# Then SVD on smaller matrix
svd = TruncatedSVD(n_components=300)
X_reduced = svd.fit_transform(X_selected)
```

**3. Mini-batch processing for very large data:**
```python
from sklearn.decomposition import IncrementalPCA

# If data doesn't fit in memory, use IncrementalPCA
# (Note: requires dense input, so convert in batches)

batch_size = 1000
ipca = IncrementalPCA(n_components=300, batch_size=batch_size)

for i in range(0, X_tfidf.shape[0], batch_size):
    batch = X_tfidf[i:i+batch_size].toarray()  # Convert batch to dense
    ipca.partial_fit(batch)

# Transform in batches
X_reduced = []
for i in range(0, X_tfidf.shape[0], batch_size):
    batch = X_tfidf[i:i+batch_size].toarray()
    X_reduced.append(ipca.transform(batch))

X_reduced = np.vstack(X_reduced)
```

**4. Alternative: Random projections (faster):**
```python
from sklearn.random_projection import SparseRandomProjection

# Very fast, theoretically sound (Johnson-Lindenstrauss)
srp = SparseRandomProjection(n_components=300, random_state=42)
X_reduced = srp.fit_transform(X_tfidf)

# Trade-off: Less optimal than SVD but much faster
```

**5. Complete pipeline:**
```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, min_df=5)),
    ('svd', TruncatedSVD(n_components=300)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Efficient processing
pipeline.fit(documents_train, y_train)
accuracy = pipeline.score(documents_test, y_test)
```

**Memory considerations:**
- **Sparse matrix:** ~50K features × 100K samples × 8 bytes = 40 GB (if dense)
- **Sparse storage:** Only non-zero values stored → typically 1-5% → 400-2000 MB
- **After SVD:** 300 × 100K × 8 bytes = 240 MB (manageable)

**Computation time:**
- **TruncatedSVD:** ~1-5 minutes for this scale
- **IncrementalPCA:** ~10-30 minutes (slower but memory-efficient)
- **Random Projection:** ~10-30 seconds (fastest)

---

## 5. Model Selection

### Definition and Fundamentals

Model selection encompasses techniques for choosing optimal models, hyperparameters, and features to maximize generalization performance. This includes cross-validation strategies for reliable performance estimation, hyperparameter tuning via grid search and randomized search, and automated model comparison. Proper model selection prevents overfitting, ensures robust evaluation, and identifies the best configuration for deployment.

### 5.1 Train-Test Split

The most basic validation strategy - splitting data into training (model fitting) and testing (performance evaluation) sets.

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Basic split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 20% for testing
    random_state=42,        # Reproducibility
    stratify=y              # Maintain class distribution
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Verify stratification
print("\nClass distribution:")
print(f"Original: {np.bincount(y) / len(y)}")
print(f"Train: {np.bincount(y_train) / len(y_train)}")
print(f"Test: {np.bincount(y_test) / len(y_test)}")

# Train and evaluate
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\nTrain accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")
```

**Three-way split (Train-Validation-Test):**
```python
# Split for hyperparameter tuning
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)  # 0.25 * 0.8 = 0.2 of total

print(f"Train: {X_train.shape[0]} (60%)")
print(f"Validation: {X_val.shape[0]} (20%)")
print(f"Test: {X_test.shape[0]} (20%)")

# Use validation for hyperparameter tuning
# Use test ONLY for final evaluation
```

### 5.2 Cross-Validation

Cross-validation provides more reliable performance estimates by training/testing on multiple data splits.

#### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.ensemble import RandomForestClassifier

# K-Fold CV (most common)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Simple cross-validation (single metric)
cv_scores = cross_val_score(rf, X, y, cv=kfold, scoring='accuracy')

print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"Individual fold scores: {cv_scores}")

# Multiple metrics evaluation
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted'
}

cv_results = cross_validate(
    rf, X, y,
    cv=kfold,
    scoring=scoring,
    return_train_score=True,
    return_estimator=False  # Set True to access fitted models
)

print(f"\nTest Accuracy: {cv_results['test_accuracy'].mean():.3f}")
print(f"Test F1: {cv_results['test_f1'].mean():.3f}")
print(f"Train-Test Gap: {cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean():.3f}")
```

#### Stratified K-Fold (for imbalanced data)

```python
from sklearn.model_selection import StratifiedKFold

# Maintains class distribution in each fold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(rf, X, y, cv=skfold, scoring='f1_weighted')

print(f"Stratified CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Verify stratification
for fold, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
    print(f"Fold {fold+1}: Train {np.bincount(y[train_idx])}, Test {np.bincount(y[test_idx])}")
```

#### Leave-One-Out CV (LOOCV)

```python
from sklearn.model_selection import LeaveOneOut

# Use when data is very small (expensive: n iterations)
loo = LeaveOneOut()

cv_scores = cross_val_score(rf, X, y, cv=loo, scoring='accuracy')

print(f"LOOCV Accuracy: {cv_scores.mean():.3f}")
print(f"Number of iterations: {len(cv_scores)}")  # Same as n_samples
```

#### Time Series CV

```python
from sklearn.model_selection import TimeSeriesSplit

# For temporal data - respects time order
tscv = TimeSeriesSplit(n_splits=5)

# Visualize splits
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold+1}:")
    print(f"  Train: {train_idx[0]} to {train_idx[-1]}")
    print(f"  Test: {test_idx[0]} to {test_idx[-1]}")

cv_scores = cross_val_score(rf, X, y, cv=tscv, scoring='accuracy')
print(f"\nTime Series CV Accuracy: {cv_scores.mean():.3f}")
```

### 5.3 Grid Search

Exhaustive search over specified hyperparameter combinations.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

# Define parameter grid
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'classifier__kernel': ['rbf', 'poly']
}

# Grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,                           # 5-fold CV
    scoring='f1_weighted',          # Optimization metric
    n_jobs=-1,                      # Parallel processing
    verbose=2,                      # Progress messages
    return_train_score=True         # Monitor overfitting
)

grid_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Test set performance
test_score = grid_search.score(X_test, y_test)
print(f"Test score: {test_score:.3f}")

# Access best model
best_model = grid_search.best_estimator_

# Analyze results
import pandas as pd

results = pd.DataFrame(grid_search.cv_results_)
results_summary = results[[
    'params', 'mean_test_score', 'std_test_score',
    'mean_train_score', 'rank_test_score'
]].sort_values('rank_test_score')

print("\nTop 5 configurations:")
print(results_summary.head())

# Check for overfitting
results['train_test_gap'] = results['mean_train_score'] - results['mean_test_score']
high_gap = results[results['train_test_gap'] > 0.1]
print(f"\nConfigurations with overfitting (gap > 0.1): {len(high_gap)}")
```

### 5.4 Randomized Search

Samples random combinations from parameter distributions - more efficient for large search spaces.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestClassifier

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 500),           # Discrete uniform
    'max_depth': [3, 5, 7, 10, 15, None],      # Choice
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)

# Randomized search
random_search = RandomizedSearchCV(
    rf,
    param_distributions,
    n_iter=100,                     # Number of random combinations
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.3f}")
print(f"Test score: {random_search.score(X_test, y_test):.3f}")

# Compare with default parameters
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
default_score = rf_default.score(X_test, y_test)

print(f"\nImprovement over default: {(random_search.score(X_test, y_test) - default_score):.3f}")
```

**Grid Search vs Randomized Search:**
```python
import time

# Grid search (exhaustive)
start = time.time()
grid_search = GridSearchCV(rf, param_grid_small, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_time = time.time() - start

# Randomized search (sampling)
start = time.time()
random_search = RandomizedSearchCV(rf, param_distributions, n_iter=50, cv=3, n_jobs=-1)
random_search.fit(X_train, y_train)
random_time = time.time() - start

print(f"Grid Search: {grid_time:.2f}s, Score: {grid_search.best_score_:.3f}")
print(f"Random Search: {random_time:.2f}s, Score: {random_search.best_score_:.3f}")
print(f"Speed-up: {grid_time/random_time:.1f}x")
```

### 5.5 Nested Cross-Validation

Unbiased performance estimation when doing hyperparameter tuning - separates model selection from evaluation.

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Nested CV: Outer loop for evaluation, inner loop for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

gb = GradientBoostingClassifier(random_state=42)

# Inner CV: Hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(gb, param_grid, cv=inner_cv, scoring='f1_weighted')

# Outer CV: Performance estimation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='f1_weighted')

print(f"Nested CV F1: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")

# Compare with non-nested (biased, overoptimistic)
non_nested_score = grid_search.fit(X, y).best_score_
print(f"Non-nested CV F1: {non_nested_score:.3f}")
print(f"Bias (overestimation): {(non_nested_score - nested_scores.mean()):.3f}")
```

### Common Pitfalls and Real-World Tips

**1. Data Leakage in Cross-Validation:**
```python
# WRONG: Fit preprocessing on entire dataset
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ❌ Uses test fold information

cv_scores = cross_val_score(model, X_scaled, y, cv=5)  # Biased!

# CORRECT: Include preprocessing in pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # ✓ Fits separately on each fold
    ('model', LogisticRegression())
])

cv_scores = cross_val_score(pipeline, X, y, cv=5)  # Unbiased
```

**2. Stratification for Imbalanced Data:**
```python
# Always use StratifiedKFold for classification
from sklearn.model_selection import StratifiedKFold

# Dataset with imbalance (90% class 0, 10% class 1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Each fold maintains 90-10 split
cv_scores = cross_val_score(model, X, y, cv=skf)
```

**3. Choosing k in K-Fold:**
```python
# Trade-offs:
# k=5: Fast, moderate bias, higher variance
# k=10: Slower, lower bias, moderate variance
# k=n (LOOCV): Slowest, lowest bias, highest variance

# Rule of thumb:
# - Small data (<1000): k=10 or LOOCV
# - Medium data (1000-10000): k=5 or k=10
# - Large data (>10000): k=3 or k=5 (or single validation set)
```

**4. Time Series Data:**
```python
# NEVER use random splits for time series
# Use TimeSeriesSplit to respect temporal order

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

# Training always precedes testing
for train_idx, test_idx in tscv.split(X):
    assert train_idx.max() < test_idx.min()  # No future leakage
```

**5. Grid Search Computational Cost:**
```python
# Number of fits = n_params × k_folds
param_grid = {
    'param1': [1, 2, 3, 4, 5],          # 5 values
    'param2': [0.1, 0.01, 0.001],       # 3 values
    'param3': ['a', 'b']                # 2 values
}
# Total combinations: 5 × 3 × 2 = 30
# With 5-fold CV: 30 × 5 = 150 model fits

# Strategy for large spaces:
# 1. Use RandomizedSearchCV (sample subset)
# 2. Coarse grid first, then fine-tune around best
# 3. Reduce cv folds (3 instead of 5)
# 4. Use n_jobs=-1 for parallelization
```

**6. Overfitting During Hyperparameter Tuning:**
```python
# Monitor train-test gap in grid search
grid_search = GridSearchCV(model, param_grid, cv=5, return_train_score=True)
grid_search.fit(X, y)

results = pd.DataFrame(grid_search.cv_results_)
results['gap'] = results['mean_train_score'] - results['mean_test_score']

# Large gaps indicate overfitting
overfit_configs = results[results['gap'] > 0.1]
print(f"Overfitting configurations: {len(overfit_configs)}")

# Choose model with good test score AND small gap
results_sorted = results.sort_values(['mean_test_score', 'gap'], 
                                     ascending=[False, True])
best_balanced = results_sorted.iloc[0]
```

**7. Final Model Evaluation:**
```python
# CRITICAL: Hold out separate test set for final evaluation

# Split 1: Train-test split (80-20)
X_temp, X_test_final, y_temp, y_test_final = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split 2: Use X_temp for grid search with CV
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_temp, y_temp)

print(f"Best CV score: {grid_search.best_score_:.3f}")

# Split 3: FINAL evaluation on held-out test set
final_score = grid_search.score(X_test_final, y_test_final)
print(f"Final test score: {final_score:.3f}")

# Never tune hyperparameters based on test set performance!
```

### Interview Questions and Answers

**Q1: Explain the difference between validation set and cross-validation. When would you use each?**

**A:**

**Validation Set:**
- Single fixed split (e.g., 60% train, 20% val, 20% test)
- Fast: Train once, evaluate once
- Higher variance: Performance depends on random split
- Less data for training

**Cross-Validation:**
- Multiple splits (e.g., 5-fold: 80% train, 20% val, repeated 5 times)
- Slower: Train k times
- Lower variance: Averaged over multiple splits
- Uses all data for training (in rotation)

**Use validation set when:**
- Large dataset (>100K samples) - single split sufficient
- Computational constraints (time-critical)
- Deep learning (expensive to train multiple times)
- Example: ImageNet classification

**Use cross-validation when:**
- Small/medium dataset (<10K samples) - need reliable estimates
- Model comparison (which algorithm performs best?)
- Hyperparameter tuning
- Example: Medical diagnosis with limited patient data

```python
# Large dataset → Validation set
if n_samples > 100000:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

# Small dataset → Cross-validation
else:
    scores = cross_val_score(model, X, y, cv=5)
    score = scores.mean()
```

**Q2: Your cross-validation score is 0.85 but test set score is 0.70. What went wrong?**

**A:** **Likely causes of this 15% gap:**

**1. Data leakage (most common):**
```python
# Problem: Preprocessing fit on entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ❌ Leakage

X_train, X_test = train_test_split(X_scaled)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # 0.85 (biased)
test_score = model.score(X_test, y_test)  # 0.70 (realistic)

# Solution: Use pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', model)
])
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
```

**2. Train-test distribution shift:**
```python
# CV on training data, test data from different distribution
# Example: Train on data from Jan-Oct, test on Nov-Dec (seasonal shift)

# Check distributions
print(f"Train mean: {X_train.mean(axis=0)}")
print(f"Test mean: {X_test.mean(axis=0)}")

# Solution: Ensure representative splits, check for temporal/geographical/demographic shifts
```

**3. Overfitting to training data:**
```python
# Model too complex for available data
# Solution: Regularization, simpler model, more data
```

**4. Small test set:**
```python
# Test set too small → high variance estimate
# If test set has only 50 samples, performance can vary significantly

# Solution: Use larger test set or cross-validation for final evaluation
```

**Debugging steps:**
```python
# 1. Check for leakage
pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
cv_with_pipeline = cross_val_score(pipeline, X_train, y_train, cv=5).mean()

# 2. Check distribution shift
from scipy.stats import ks_2samp
for i in range(X.shape[1]):
    stat, p = ks_2samp(X_train[:, i], X_test[:, i])
    if p < 0.05:
        print(f"Feature {i}: Distribution differs (p={p:.4f})")

# 3. Use nested CV for unbiased estimate
```

**Q3: How do you choose between Grid Search and Randomized Search?**

**A:**

**Grid Search:**
- **Exhaustive:** Tests all combinations
- **Use when:** Small search space (<100 combinations), need guarantee of finding best
- **Example:** 3 parameters with 3-5 values each

```python
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1],
    'kernel': ['rbf', 'poly']
}
# Total: 3 × 3 × 2 = 18 combinations (manageable)
```

**Randomized Search:**
- **Sampling:** Tests random subset
- **Use when:** Large search space (>1000 combinations), continuous parameters, time-constrained
- **Example:** Many parameters with wide ranges

```python
from scipy.stats import uniform, randint

param_dist = {
    'n_estimators': randint(50, 500),          # 450 possibilities
    'max_depth': randint(3, 20),               # 17 possibilities
    'min_samples_split': randint(2, 50),       # 48 possibilities
    'learning_rate': uniform(0.001,0.5)  # Continuous
}
# Total: ~450 × 17 × 48 × ∞ = millions of combinations
# Sample only 100-200 with RandomizedSearchCV
```

**Decision Framework:**

| Criterion | Grid Search | Randomized Search |
|-----------|-------------|-------------------|
| **Search space size** | <100 combinations | >1000 combinations |
| **Parameter types** | Discrete choices | Continuous/large discrete |
| **Time budget** | Sufficient for exhaustive | Limited time |
| **Guarantee** | Finds true best in grid | May miss optimal |
| **Efficiency** | Tests redundant combos | Explores space efficiently |

**Practical approach:**
```python
# 1. Start with Randomized Search (broad exploration)
random_search = RandomizedSearchCV(
    model, param_distributions, 
    n_iter=100, cv=3, random_state=42
)
random_search.fit(X_train, y_train)

# 2. Identify promising region
best_params = random_search.best_params_

# 3. Fine-tune with Grid Search (narrow refinement)
refined_grid = {
    'n_estimators': [best_params['n_estimators'] - 20, 
                     best_params['n_estimators'],
                     best_params['n_estimators'] + 20],
    'learning_rate': [best_params['learning_rate'] * 0.5,
                      best_params['learning_rate'],
                      best_params['learning_rate'] * 1.5]
}

grid_search = GridSearchCV(model, refined_grid, cv=5)
grid_search.fit(X_train, y_train)
```

**Q4: Explain nested cross-validation. When is it necessary?**

**A:** **Nested CV provides unbiased performance estimates when doing hyperparameter tuning.**

**Structure:**
- **Outer loop:** Evaluates model generalization (5-10 folds)
- **Inner loop:** Selects hyperparameters (3-5 folds)

```python
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold

# Inner CV: Hyperparameter selection
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='f1')

# Outer CV: Performance estimation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='f1')

print(f"Nested CV score: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
```

**Why necessary:**

**Without nested CV (biased):**
```python
# This is WRONG for reporting model performance
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)
print(f"Best score: {grid_search.best_score_:.3f}")  # Overoptimistic!

# Problem: Hyperparameters selected to maximize this exact data
# Like "peeking" at test set during training
```

**With nested CV (unbiased):**
```python
# Each outer fold uses DIFFERENT data for:
# - Inner CV: Selecting hyperparameters
# - Evaluation: Testing selected model

# No information leakage between selection and evaluation
```

**When to use:**
- **Research:** Reporting model performance in papers
- **Small datasets:** Can't afford separate test set
- **Model comparison:** Which algorithm is best?
- **Production:** Estimating expected performance

**When NOT needed:**
- **Large datasets:** Can use separate validation and test sets
- **Just hyperparameter tuning:** Regular GridSearchCV sufficient if you have held-out test set
- **Time constraints:** Nested CV is computationally expensive (k_outer × k_inner fits)

**Computational cost:**
```python
# Regular CV: k folds
# Nested CV: k_outer × k_inner folds

# Example: 5-fold outer, 3-fold inner, 20 param combinations
# Regular CV: 20 × 3 = 60 fits
# Nested CV: 5 × (20 × 3) = 300 fits (5× slower)
```

**Q5: How do you handle time series data in cross-validation?**

**A:** **Time series requires special CV strategies that respect temporal ordering** - cannot use random splits.

**Key principle:** Training data must always precede test data (no future leakage).

**Method 1: Time Series Split (Expanding Window):**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

# Visualization of splits
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold+1}:")
    print(f"  Train: indices {train_idx[0]} to {train_idx[-1]}")
    print(f"  Test:  indices {test_idx[0]} to {test_idx[-1]}")

# Example output:
# Fold 1: Train [0-199], Test [200-299]
# Fold 2: Train [0-299], Test [300-399]
# Fold 3: Train [0-399], Test [400-499]
# Training set grows each fold

cv_scores = cross_val_score(model, X, y, cv=tscv)
```

**Method 2: Sliding Window:**
```python
from sklearn.model_selection import TimeSeriesSplit

# Custom sliding window implementation
def sliding_window_cv(X, y, window_size=100, test_size=20, step=20):
    """Fixed-size training window slides forward"""
    n = len(X)
    scores = []
    
    for start in range(0, n - window_size - test_size, step):
        train_end = start + window_size
        test_end = train_end + test_size
        
        X_train = X[start:train_end]
        y_train = y[start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return np.array(scores)

scores = sliding_window_cv(X, y, window_size=500, test_size=100)
print(f"Sliding window CV: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Method 3: Blocked CV with Gap:**
```python
# Add gap between train and test to prevent leakage
# Useful when predictions affect future (e.g., algorithmic trading)

def blocked_cv_with_gap(X, y, n_splits=5, gap=10):
    """Add gap between training and test sets"""
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        # Remove last 'gap' samples from training
        train_idx = train_idx[:-gap]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return np.array(scores)
```

**Common mistakes to avoid:**
```python
# ❌ WRONG: Random K-Fold for time series
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)  # Uses future data in training!

# ❌ WRONG: Sorting before split
X_sorted = X[np.argsort(timestamps)]
X_train, X_test = train_test_split(X_sorted)  # Still random split!

# ✓ CORRECT: Use TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(model, X, y, cv=tscv)
```

**Additional considerations:**
```python
# 1. Feature engineering respects time
# Don't use future information in features (lag features only)

# 2. Handle seasonality
# Ensure test set covers all seasons if relevant

# 3. Walk-forward validation for final model
# Retrain on all data up to deployment date
```

**Q6: You have an imbalanced dataset (95% class 0, 5% class 1). How do you set up proper validation?**

**A:** **Imbalanced data requires stratified splitting and appropriate evaluation metrics.**

**1. Stratified splits (critical):**
```python
from sklearn.model_selection import train_test_split, StratifiedKFold

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintains 95-5 split in both sets
)

# Verify stratification
print(f"Train distribution: {np.bincount(y_train) / len(y_train)}")
print(f"Test distribution: {np.bincount(y_test) / len(y_test)}")

# Cross-validation with stratification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

**2. Appropriate metrics (not accuracy):**
```python
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, average_precision_score
)

# Accuracy is misleading (predicting all 0s gives 95% accuracy!)
accuracy = model.score(X_test, y_test)  # Don't rely on this

# Better metrics
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")  # Harmonic mean of precision/recall
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")  # Threshold-independent
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"PR-AUC: {average_precision_score(y_test, y_proba):.3f}")  # Better for imbalance
```

**3. Cross-validation with proper scoring:**
```python
# Use F1 or PR-AUC for optimization
cv_scores = cross_val_score(
    model, X, y, 
    cv=skf, 
    scoring='f1'  # Or 'average_precision', 'roc_auc'
)

# Or evaluate multiple metrics
from sklearn.model_selection import cross_validate

scoring = {
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'roc_auc': 'roc_auc'
}

cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring)

print(f"F1: {cv_results['test_f1'].mean():.3f}")
print(f"ROC-AUC: {cv_results['test_roc_auc'].mean():.3f}")
```

**4. Handle class imbalance:**
```python
# Option A: Class weights
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')  # Automatically adjusts

# Option B: Resampling (but within CV folds!)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),  # Applies only to training fold
    ('classifier', LogisticRegression())
])

cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='f1')
```

**5. Monitor both classes:**
```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Output shows per-class performance:
#               precision    recall  f1-score   support
# Negative         0.98      0.99      0.98       190
# Positive         0.75      0.60      0.67        10
```

**Q7: How do you prevent data leakage in a machine learning pipeline?**

**A:** **Data leakage occurs when information from test/validation data influences training, leading to overoptimistic performance estimates.**

**Common sources and solutions:**

**1. Preprocessing leakage (most common):**
```python
# ❌ WRONG: Fit on entire dataset
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses test set statistics!

X_train, X_test = train_test_split(X_scaled)
model.fit(X_train, y_train)
# Test score will be overoptimistic

# ✓ CORRECT: Use Pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Fits only on training data
    ('model', LogisticRegression())
])

X_train, X_test = train_test_split(X, y)
pipeline.fit(X_train, y_train)
# Test score is unbiased
```

**2. Feature engineering leakage:**
```python
# ❌ WRONG: Target encoding using entire dataset
from category_encoders import TargetEncoder

encoder = TargetEncoder()
X_encoded = encoder.fit_transform(X, y)  # Uses test targets!

# ✓ CORRECT: Include in pipeline
pipeline = Pipeline([
    ('encoder', TargetEncoder()),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)  # Encodes using training targets only
```

**3. Temporal leakage:**
```python
# ❌ WRONG: Features that include future information
# Example: "days_until_next_purchase" - only known in hindsight

# ✓ CORRECT: Use only past information
# "days_since_last_purchase", "average_purchase_frequency"

# For time series: Use lag features only
df['feature_lag1'] = df['feature'].shift(1)  # Previous day
df['feature_lag7'] = df['feature'].shift(7)  # Week ago
```

**4. Duplicate data:**
```python
# ❌ Problem: Same user in train and test
# Model memorizes user patterns instead of generalizing

# ✓ Solution: Split by user/group
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=user_ids))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
```

**5. Feature selection leakage:**
```python
# ❌ WRONG: Select features using entire dataset
from sklearn.feature_selection import SelectKBest

selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # Uses test data!

# ✓ CORRECT: Include in pipeline
pipeline = Pipeline([
    ('selector', SelectKBest(k=10)),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

**6. Outlier removal leakage:**
```python
# ❌ WRONG: Identify outliers using entire dataset
from sklearn.ensemble import IsolationForest

iso = IsolationForest()
outliers = iso.fit_predict(X)  # Uses test data patterns!
X_clean = X[outliers == 1]

# ✓ CORRECT: Identify outliers on training set only
X_train, X_test, y_train, y_test = train_test_split(X, y)

iso = IsolationForest()
outliers_train = iso.fit_predict(X_train)
X_train_clean = X_train[outliers_train == 1]
y_train_clean = y_train[outliers_train == 1]

# Don't remove outliers from test set (real-world will have them)
```

**7. Cross-validation with Pipeline:**
```python
# ✓ Best practice: Everything in pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('feature_selector', SelectKBest(k=20)),
    ('classifier', RandomForestClassifier())
])

# All preprocessing happens within each CV fold
cv_scores = cross_val_score(pipeline, X, y, cv=5)
# No leakage possible
```

**Checklist to prevent leakage:**
- [ ] All preprocessing in Pipeline or done separately per fold
- [ ] Train-test split before any data exploration or transformation
- [ ] No information from target variable in features (except supervised transformations in pipeline)
- [ ] Time series: Only past information in features
- [ ] Group-aware splitting when needed (users, hospitals, etc.)
- [ ] Same preprocessing applied to train and test (fitted on train only)

**Q8: What's the difference between model evaluation and model selection? How does this affect your validation strategy?**

**A:** **Model evaluation estimates generalization performance; model selection chooses best hyperparameters/algorithm.**

**Key distinction:**

**Model Selection (hyperparameter tuning):**
- **Goal:** Find best configuration
- **Method:** Grid/Random search with cross-validation
- **Uses:** Training + validation data
- **Output:** Optimized model

**Model Evaluation (performance estimation):**
- **Goal:** Estimate real-world performance
- **Method:** Test on held-out data (never used for tuning)
- **Uses:** Test data only
- **Output:** Unbiased performance estimate

**Three-way split strategy:**
```python
# 1. Split data three ways
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Result: 60% train, 20% validation, 20% test

# 2. Model Selection: Use train + validation
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)  # CV uses train data

# Validate
val_score = grid_search.score(X_val, y_val)

# 3. Model Evaluation: Use test (only once!)
test_score = grid_search.score(X_test, y_test)

print(f"Validation score (for tuning): {val_score:.3f}")
print(f"Test score (final estimate): {test_score:.3f}")

# NEVER tune based on test score!
```

**Alternative: CV for selection, held-out for evaluation:**
```python
# 1. Split once: train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Model Selection: CV on training set
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best CV score (selection): {grid_search.best_score_:.3f}")

# 3. Model Evaluation: Test set (once!)
test_score = grid_search.score(X_test, y_test)
print(f"Test score (evaluation): {test_score:.3f}")
```

**Why this matters:**
```python
# ❌ WRONG: Tuning on test set
param_grid = {'C': [0.1, 1, 10]}

best_score = 0
best_C = None

for C in param_grid['C']:
    model = LogisticRegression(C=C)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # ❌ Using test for selection!
    
    if score > best_score:
        best_score = score
        best_C = C

# Problem: Test score is now optimistic (you "peeked")
print(f"Best C: {best_C}, Score: {best_score}")  # Biased!

# ✓ CORRECT: Never touch test during selection
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)  # Selection uses only train

test_score = grid_search.score(X_test, y_test)  # Evaluation uses test once
print(f"Unbiased test score: {test_score:.3f}")
```

**Golden rule:** Test set is opened only once, after all model selection is complete.

---

## 6. Preprocessing

### Definition and Fundamentals

Preprocessing transforms raw data into a format suitable for machine learning algorithms. Scikit-learn provides transformers for handling missing values, scaling features, encoding categorical variables, and engineering new features. Proper preprocessing is critical for model performance - distance-based algorithms require scaling, tree-based methods handle raw features naturally, and all models benefit from proper handling of missing values and categorical encoding.

### 6.1 Handling Missing Values

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import pandas as pd

# Create dataset with missing values
X = np.array([[1, 2], [np.nan, 3], [7, 6], [5, np.nan]])

# Strategy 1: Simple Imputation
imputer_mean = SimpleImputer(strategy='mean')  # 'mean', 'median', 'most_frequent', 'constant'
X_imputed_mean = imputer_mean.fit_transform(X)

print("Mean imputation:")
print(X_imputed_mean)

# Strategy 2: KNN Imputation (uses neighbors)
knn_imputer = KNNImputer(n_neighbors=2, weights='uniform')
X_imputed_knn = knn_imputer.fit_transform(X)

print("\nKNN imputation:")
print(X_imputed_knn)

# Strategy 3: Iterative Imputation (MICE - Multiple Imputation)
iter_imputer = IterativeImputer(max_iter=10, random_state=42)
X_imputed_iter = iter_imputer.fit_transform(X)

print("\nIterative imputation:")
print(X_imputed_iter)

# Strategy 4: Constant value
imputer_const = SimpleImputer(strategy='constant', fill_value=-999)
X_imputed_const = imputer_const.fit_transform(X)
```

**Missing value indicator:**
```python
from sklearn.impute import MissingIndicator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Create binary features indicating which values were missing
indicator = MissingIndicator()
X_indicator = indicator.fit_transform(X)

print("Missing indicators:")
print(X_indicator)  # True where values were missing

# Combine imputation with indicator features
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('add_indicator', MissingIndicator())
])
```

### 6.2 Feature Scaling

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, Normalizer
)

X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

# StandardScaler: mean=0, std=1 (most common)
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X)
print("StandardScaler:")
print(f"Mean: {X_standard.mean(axis=0)}")
print(f"Std: {X_standard.std(axis=0)}")

# MinMaxScaler: scale to [0, 1] range
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
print("\nMinMaxScaler:")
print(f"Min: {X_minmax.min(axis=0)}")
print(f"Max: {X_minmax.max(axis=0)}")

# RobustScaler: uses median and IQR (robust to outliers)
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)
print("\nRobustScaler (good for outliers):")
print(X_robust)

# MaxAbsScaler: scale by maximum absolute value (preserves sparsity)
scaler_maxabs = MaxAbsScaler()
X_maxabs = scaler_maxabs.fit_transform(X)
print("\nMaxAbsScaler:")
print(X_maxabs)

# Normalizer: scale samples to unit norm (L1 or L2)
normalizer = Normalizer(norm='l2')  # 'l1', 'l2', 'max'
X_normalized = normalizer.fit_transform(X)
print("\nNormalizer (per sample):")
print(f"L2 norms: {np.linalg.norm(X_normalized, axis=1)}")
```

**When to use which scaler:**
```python
# StandardScaler: Default choice, assumes Gaussian distribution
# - Use for: Linear models, SVM, KNN, Neural Networks

# MinMaxScaler: When you need bounded range [0, 1]
# - Use for: Neural networks (certain activations), image data

# RobustScaler: When data has outliers
# - Use for: Data with extreme values that shouldn't dominate

# MaxAbsScaler: For sparse data (preserves zeros)
# - Use for: Sparse matrices, text data already scaled

# Normalizer: When direction matters more than magnitude
# - Use for: Text classification (TF-IDF), clustering
```

### 6.3 Encoding Categorical Variables

```python
from sklearn.preprocessing import (
    OrdinalEncoder, OneHotEncoder, LabelEncoder
)

# Sample categorical data
categories = np.array([['red', 'small'],
                      ['blue', 'medium'],
                      ['green', 'large'],
                      ['red', 'medium']])

# OrdinalEncoder: For ordinal categories (order matters)
ordinal_encoder = OrdinalEncoder(categories=[['small', 'medium', 'large']])
# Note: This encodes ALL features, select columns carefully

# OneHotEncoder: For nominal categories (no order)
onehot_encoder = OneHotEncoder(
    sparse_output=False,      # Return dense array
    handle_unknown='ignore',  # Ignore unknown categories during transform
    drop='first'              # Drop first category to avoid multicollinearity
)

X_onehot = onehot_encoder.fit_transform(categories)
print("One-Hot Encoded:")
print(X_onehot)
print("Feature names:", onehot_encoder.get_feature_names_out())

# LabelEncoder: For target variable only (not features!)
label_encoder = LabelEncoder()
labels = ['cat', 'dog', 'bird', 'cat', 'dog']
y_encoded = label_encoder.fit_transform(labels)
print("\nLabel Encoded:")
print(y_encoded)
print("Classes:", label_encoder.classes_)

# Inverse transform
y_decoded = label_encoder.inverse_transform(y_encoded)
print("Decoded:", y_decoded)
```

**Handling high-cardinality categoricals:**
```python
# Target encoding for high-cardinality features
from category_encoders import TargetEncoder

# Example: city names (1000+ unique values)
# One-hot would create 1000 features!

# Target encoding: Replace category with mean of target
target_encoder = TargetEncoder()
# Must be used in pipeline to avoid leakage!

# Frequency encoding
df['city_frequency'] = df['city'].map(df['city'].value_counts())

# Hash encoding (fixed dimensionality)
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=10, input_type='string')
```

---

## 7. Model Evaluation

### Definition and Fundamentals

Model evaluation measures how well a model generalizes to unseen data. Scikit-learn provides comprehensive metrics for classification (accuracy, precision, recall, F1, ROC-AUC) and regression (MSE, MAE, R²), along with tools for confusion matrices, ROC curves, and learning curves. Choosing appropriate metrics is crucial - accuracy misleads on imbalanced data, while metrics like F1 and PR-AUC better capture performance trade-offs.

### 7.1 Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, log_loss
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                          random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("[[TN, FP],")
print(" [FN, TP]]")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC: {roc_auc:.3f}")

# PR-AUC (better for imbalanced data)
pr_auc = average_precision_score(y_test, y_proba)
print(f"PR-AUC: {pr_auc:.3f}")

# Log Loss (probabilistic performance)
logloss = log_loss(y_test, y_proba)
print(f"Log Loss: {logloss:.3f}")
```

**ROC and Precision-Recall Curves:**
```python
# ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)

# Precision-Recall Curve
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_proba)

plt.subplot(1, 2, 2)
plt.plot(recall_curve, precision_curve, label=f'PR Curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('roc_pr_curves.png')
```

**Optimal Threshold Selection:**
```python
# Find threshold that maximizes F1 score
from sklearn.metrics import f1_score

thresholds = np.linspace(0, 1, 100)
f1_scores = []

for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thresh))

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F1 at optimal threshold: {f1_scores[optimal_idx]:.3f}")

# Apply optimal threshold
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
print(f"Precision: {precision_score(y_test, y_pred_optimal):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_optimal):.3f}")
```

**Multiclass Classification Metrics:**
```python
from sklearn.datasets import load_iris
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

rf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
rf_multi.fit(X_train, y_train)
y_pred_multi = rf_multi.predict(X_test)
y_proba_multi = rf_multi.predict_proba(X_test)

# Macro vs Micro vs Weighted averaging
print("Multiclass Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_multi):.3f}")
print(f"F1 (macro): {f1_score(y_test, y_pred_multi, average='macro'):.3f}")
print(f"F1 (micro): {f1_score(y_test, y_pred_multi, average='micro'):.3f}")
print(f"F1 (weighted): {f1_score(y_test, y_pred_multi, average='weighted'):.3f}")

# Cohen's Kappa (accounts for chance agreement)
kappa = cohen_kappa_score(y_test, y_pred_multi)
print(f"Cohen's Kappa: {kappa:.3f}")

# Multiclass ROC-AUC (One-vs-Rest)
roc_auc_ovr = roc_auc_score(y_test, y_proba_multi, multi_class='ovr')
print(f"ROC-AUC (OvR): {roc_auc_ovr:.3f}")

# Confusion matrix visualization
import seaborn as sns

cm_multi = confusion_matrix(y_test, y_pred_multi)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_multiclass.png')
```

### 7.2 Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    explained_variance_score, max_error
)
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

# Load regression dataset
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Train model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

# Common metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print("Regression Metrics:")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R² Score: {r2:.3f}")
print(f"MAPE: {mape:.3%}")
print(f"Median AE: {medae:.3f}")
print(f"Explained Variance: {evs:.3f}")
print(f"Max Error: {max_error(y_test, y_pred):.3f}")
```

**Residual Analysis:**
```python
# Residual plot
residuals = y_test - y_pred

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Predicted
axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residual Plot')

# Histogram of residuals
axes[0, 1].hist(residuals, bins=30, edgecolor='black')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Residual Distribution')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')

# Actual vs Predicted
axes[1, 1].scatter(y_test, y_pred, alpha=0.6)
axes[1, 1].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual Values')
axes[1, 1].set_ylabel('Predicted Values')
axes[1, 1].set_title('Actual vs Predicted')

plt.tight_layout()
plt.savefig('residual_analysis.png')
```

### 7.3 Cross-Validation Metrics

```python
from sklearn.model_selection import cross_val_score, cross_validate

# Single metric
cv_scores = cross_val_score(ridge, X_train, y_train, 
                            cv=5, scoring='r2')
print(f"CV R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Multiple metrics
scoring = {
    'r2': 'r2',
    'neg_mse': 'neg_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error'
}

cv_results = cross_validate(ridge, X_train, y_train,
                            cv=5, scoring=scoring,
                            return_train_score=True)

print("\nCross-Validation Results:")
print(f"Test R²: {cv_results['test_r2'].mean():.3f}")
print(f"Test MSE: {-cv_results['test_neg_mse'].mean():.3f}")
print(f"Test MAE: {-cv_results['test_neg_mae'].mean():.3f}")
print(f"Train R²: {cv_results['train_r2'].mean():.3f}")
print(f"Overfitting gap: {cv_results['train_r2'].mean() - cv_results['test_r2'].mean():.3f}")
```

### 7.4 Custom Scoring Functions

```python
from sklearn.metrics import make_scorer

# Define custom metric
def custom_metric(y_true, y_pred):
    """Custom metric: penalize overestimation more than underestimation"""
    errors = y_true - y_pred
    overestimation = errors[errors < 0]
    underestimation = errors[errors >= 0]
    
    return np.mean(np.abs(underestimation)) + 2 * np.mean(np.abs(overestimation))

# Create scorer (greater_is_better=False because lower error is better)
custom_scorer = make_scorer(custom_metric, greater_is_better=False)

# Use in cross-validation
cv_custom = cross_val_score(ridge, X_train, y_train, 
                           cv=5, scoring=custom_scorer)
print(f"Custom metric CV score: {cv_custom.mean():.3f}")

# Use in GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(Ridge(), param_grid, 
                          cv=5, scoring=custom_scorer)
grid_search.fit(X_train, y_train)
```

### Common Pitfalls and Real-World Tips

**1. Metric Selection for Imbalanced Data:**
```python
# Imbalanced dataset (95% class 0, 5% class 1)
# Accuracy is misleading!

# Bad metric
accuracy = accuracy_score(y_test, y_pred)  # Might be 95% by predicting all 0s!

# Better metrics
f1 = f1_score(y_test, y_pred)              # Balances precision and recall
pr_auc = average_precision_score(y_test, y_proba)  # Best for imbalance
roc_auc = roc_auc_score(y_test, y_proba)   # Threshold-independent

# Class-specific metrics
report = classification_report(y_test, y_pred, output_dict=True)
minority_recall = report['1']['recall']     # Focus on minority class
```

**2. Understanding Precision-Recall Trade-off:**
```python
# High precision: Few false positives (conservative predictions)
# High recall: Few false negatives (liberal predictions)

# Spam detection: Prefer high precision (avoid false positives)
# Cancer screening: Prefer high recall (catch all cases)

# Adjust threshold based on business needs
threshold_high_precision = 0.8  # Fewer positives, more confident
threshold_high_recall = 0.3     # More positives, catch more cases

y_pred_conservative = (y_proba >= threshold_high_precision).astype(int)
y_pred_liberal = (y_proba >= threshold_high_recall).astype(int)
```

**3. R² Can Be Negative:**
```python
# R² < 0 means model performs worse than mean baseline
# This happens when model is very poor or test data differs from train

# Example
y_test_shifted = y_test + 100  # Simulate distribution shift
r2_negative = r2_score(y_test_shifted, y_pred)
print(f"R² on shifted data: {r2_negative:.3f}")  # Likely negative

# Solution: Check for data leakage, distribution shift, or model issues
```

**4. MSE vs MAE Trade-offs:**
```python
# MSE: Penalizes large errors more (squared term)
# - Use when: Large errors are particularly bad
# - Sensitive to outliers

# MAE: Treats all errors equally
# - Use when: All errors equally important
# - Robust to outliers

# Example
errors = np.array([1, 1, 1, 10])  # One large error

mse = np.mean(errors ** 2)  # = 26.0
mae = np.mean(errors)        # = 3.25

print(f"MSE: {mse:.2f}")  # Heavily influenced by outlier
print(f"MAE: {mae:.2f}")  # Less affected
```

**5. Stratification in Classification:**
```python
# Always stratify for imbalanced classification
from sklearn.model_selection import train_test_split, StratifiedKFold

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf)
```

**6. Macro vs Micro vs Weighted Averaging:**
```python
# Multiclass with imbalanced classes: [100, 50, 10] samples

# Macro: Average metrics per class (equal weight per class)
# Good for: Treating all classes equally
f1_macro = f1_score(y_test, y_pred, average='macro')

# Micro: Calculate metrics globally (weighted by support)
# Good for: Overall performance across all samples
f1_micro = f1_score(y_test, y_pred, average='micro')

# Weighted: Average weighted by class support
# Good for: Imbalanced datasets
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"Macro F1: {f1_macro:.3f}")      # Sensitive to minority class
print(f"Micro F1: {f1_micro:.3f}")      # Overall accuracy equivalent
print(f"Weighted F1: {f1_weighted:.3f}")  # Balanced for imbalance
```

### Interview Questions and Answers

**Q1: When would you use Precision over Recall, and vice versa? Give real-world examples.**

**A:**

**Use Precision when false positives are costly:**
- **Spam detection:** False positive = legitimate email in spam (user misses important message)
- **Marketing campaigns:** False positive = waste money on uninterested customers
- **Fraud detection alerts:** Too many false alarms → alert fatigue
- **Medical surgery recommendations:** False positive = unnecessary surgery

**Use Recall when false negatives are costly:**
- **Cancer screening:** False negative = missed diagnosis (potentially fatal)
- **Security threat detection:** False negative = breach goes undetected
- **Loan default prediction:** False negative = lend to someone who will default
- **Manufacturing defect detection:** False negative = defective product shipped

**Balanced approach (F1 score):**
- When false positives and false negatives are equally important
- General classification tasks without specific cost considerations

**Practical implementation:**
```python
# Adjust threshold based on cost function
costs = {
    'false_positive_cost': 10,   # Cost of FP
    'false_negative_cost': 100   # Cost of FN (much worse)
}

# Optimize threshold to minimize total cost
thresholds = np.linspace(0, 1, 100)
costs_list = []

for thresh in thresholds:
    y_pred = (y_proba >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total_cost = (fp * costs['false_positive_cost'] + 
                 fn * costs['false_negative_cost'])
    costs_list.append(total_cost)

optimal_threshold = thresholds[np.argmin(costs_list)]
print(f"Optimal threshold for cost minimization: {optimal_threshold:.3f}")
```

**Q2: Explain why accuracy is misleading for imbalanced datasets. What metrics should you use instead?**

**A:**

**Why accuracy is misleading:**
```python
# Dataset: 95% class 0, 5% class 1
# Naive model: Always predict class 0

y_test = np.array([0]*95 + [1]*5)
y_pred_naive = np.array([0]*100)  # Always predict 0

accuracy = accuracy_score(y_test, y_pred_naive)
print(f"Accuracy: {accuracy:.1%}")  # 95%! Looks great but useless!

# Problem: High accuracy despite never detecting minority class
recall_class1 = recall_score(y_test, y_pred_naive)
print(f"Recall for class 1: {recall_class1:.1%}")  # 0%!
```

**Better metrics for imbalanced data:**

**1. F1 Score (harmonic mean of precision and recall):**
```python
f1 = f1_score(y_test, y_pred)
# Balances precision and recall
# Low if either is poor
```

**2. PR-AUC (Area Under Precision-Recall Curve):**
```python
pr_auc = average_precision_score(y_test, y_proba)
# Better than ROC-AUC for imbalanced data
# Focuses on positive class performance
```

**3. ROC-AUC (with caution):**
```python
roc_auc = roc_auc_score(y_test, y_proba)
# Threshold-independent
# Can be optimistic for severe imbalance
```

**4. Class-specific metrics:**
```python
# Focus on minority class performance
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred, output_dict=True)
minority_precision = report['1']['precision']
minority_recall = report['1']['recall']
minority_f1 = report['1']['f1-score']
```

**5. Balanced Accuracy:**
```python
from sklearn.metrics import balanced_accuracy_score

# Average of recall per class (equal weight to each class)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
```

**6. Cohen's Kappa:**
```python
from sklearn.metrics import cohen_kappa_score

# Accounts for chance agreement
kappa = cohen_kappa_score(y_test, y_pred)
```

**Q3: How do you interpret a confusion matrix? Walk me through an example.**

**A:**

**Confusion Matrix Structure:**
```
                Predicted
              Negative  Positive
Actual  Neg      TN        FP     (Specificity = TN/(TN+FP))
        Pos      FN        TP     (Recall = TP/(TP+FN))
                 
            (Precision = TP/(TP+FP))
```

**Example: Medical diagnosis**
```python
# 100 patients tested for disease
# 10 actually have disease, 90 don't

y_true = np.array([0]*90 + [1]*10)
y_pred = np.array([0]*85 + [1]*5 + [0]*2 + [1]*8)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
# [[85  5]   <- 85 correctly identified as healthy (TN),
#  [ 2  8]]      5 healthy wrongly flagged (FP)
#               2 sick missed (FN), 8 sick caught (TP)

# Interpretation:
tn, fp, fn, tp = cm.ravel()

# True Negatives (85): Correctly identified healthy patients
# False Positives (5): Healthy patients incorrectly flagged as sick
# False Negatives (2): Sick patients missed (CRITICAL!)
# True Positives (8): Sick patients correctly identified

# Derived metrics:
sensitivity = tp / (tp + fn)  # = 8/10 = 0.8 (Recall)
specificity = tn / (tn + fp)  # = 85/90 = 0.944
precision = tp / (tp + fp)     # = 8/13 = 0.615
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

print(f"\nSensitivity (Recall): {sensitivity:.1%}")  # Caught 80% of sick patients
print(f"Specificity: {specificity:.1%}")             # Correctly identified 94.4% of healthy
print(f"Precision: {precision:.1%}")                 # 61.5% of positive predictions correct
print(f"F1 Score: {f1:.3f}")
```

**Business implications:**
- **2 False Negatives:** Missed 2 sick patients (could be fatal)
- **5 False Positives:** 5 healthy patients undergo unnecessary tests (costly, stressful)
- **Trade-off decision:** Lower threshold to catch more sick patients (increase recall), but more false alarms

**Q4: What's the difference between ROC-AUC and PR-AUC? When would you use each?**

**A:**

**ROC-AUC (Receiver Operating Characteristic):**
- **X-axis:** False Positive Rate = FP / (FP + TN)
- **Y-axis:** True Positive Rate (Recall) = TP / (TP + FN)
- **Range:** 0.5 (random) to 1.0 (perfect)
- **Interpretation:** Probability that model ranks random positive higher than random negative

**PR-AUC (Precision-Recall):**
- **X-axis:** Recall = TP / (TP + FN)
- **Y-axis:** Precision = TP / (TP + FP)
- **Range:** Baseline = % positives in data to 1.0 (perfect)
- **Interpretation:** Trade-off between precision and recall at different thresholds

**Key differences:**
```python
# Imbalanced dataset: 95% class 0, 5% class 1

# ROC-AUC can be misleadingly high
# - FPR denominator includes many negatives (TN)
# - Small number of FP has little impact on FPR

# PR-AUC more informative
# - Precision denominator = TP + FP
# - Directly shows positive class performance
```

**Use ROC-AUC when:**
- Balanced datasets
- Both classes equally important
- Comparing multiple models
- Example: Binary classification with 50-50 split

**Use PR-AUC when:**
- **Imbalanced datasets** (critical difference!)
- Focus on positive class performance
- Cost of false positives high relative to true negatives
- Example: Fraud detection (1% fraud), anomaly detection, rare disease diagnosis

**Practical example:**
```python
from sklearn.metrics import roc_auc_score, average_precision_score

# Severely imbalanced: 99% class 0, 1% class 1
y_imbalanced = np.array([0]*990 + [1]*10)
y_proba = np.random.rand(1000)

roc_auc = roc_auc_score(y_imbalanced, y_proba)
pr_auc = average_precision_score(y_imbalanced, y_proba)

print(f"ROC-AUC: {roc_auc:.3f}")  # Might be ~0.7 (looks decent)
print(f"PR-AUC: {pr_auc:.3f}")    # Might be ~0.05 (reveals poor performance)

# PR-AUC better reflects true performance on minority class
```

**Q5: Explain R² score. Can it be negative? What does that mean?**

**A:**

**R² (Coefficient of Determination):**
```
R² = 1 - (SS_residual / SS_total)
   = 1 - (Σ(y_true - y_pred)² / Σ(y_true - y_mean)²)
```

**Interpretation:**
- **R² = 1.0:** Perfect predictions (all variance explained)
- **R² = 0.0:** Model performs same as mean baseline
- **R² < 0.0:** Model performs WORSE than predicting mean
- **R² = 0.5:** Model explains 50% of variance

**Yes, R² can be negative:**
```python
# Example: Model is terrible
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([10, 11, 12, 13, 14])  # Completely wrong

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")  # Negative!

# Baseline (mean)
y_mean = np.full_like(y_true, y_true.mean())
mse_model = mean_squared_error(y_true, y_pred)
mse_baseline = mean_squared_error(y_true, y_mean)

print(f"Model MSE: {mse_model:.2f}")
print(f"Baseline MSE: {mse_baseline:.2f}")
# Model worse than baseline → R² < 0
```

**When R² is negative:**
1. **Severe overfitting:** Model memorized training data, fails on test
2. **Data distribution shift:** Test data from different distribution than train
3. **Data leakage:** Leakage in training but not in test
4. **Wrong model choice:** Model assumptions violated
5. **Extrapolation:** Test data outside training range (trees can't extrapolate)

**Limitations of R²:**
- Not suitable for non-linear relationships without transformation
- Sensitive to outliers
- Always increases with more features (use adjusted R² for feature selection)
- Doesn't indicate if model is biased

**Adjusted R²:**
```python
# Penalizes additional features
def adjusted_r2(r2, n_samples, n_features):
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

r2 = r2_score(y_test, y_pred)
adj_r2 = adjusted_r2(r2, len(y_test), X_test.shape[1])

print(f"R²: {r2:.3f}")
print(f"Adjusted R²: {adj_r2:.3f}")
```

---

This completes the comprehensive Scikit-Learn reference guide covering the seven core topics relevant for Data Science and ML engineering interviews (entry to 1-3 years experience). The guide includes:

- **Classification:** Logistic Regression, SVM, Decision Trees, Random Forests, Gradient Boosting, KNN, Naive Bayes
- **Regression:** Linear, Ridge, Lasso, ElasticNet, Tree-based, SVR, KNN
- **Clustering:** K-Means, DBSCAN, Hierarchical, GMM
- **Dimensionality Reduction:** PCA, Kernel PCA, TruncatedSVD, t-SNE
- **Model Selection:** Train-test split, Cross-validation, Grid/Random Search, Nested CV
- **Preprocessing:** Missing values, scaling, encoding
- **Model Evaluation:** Classification and regression metrics, confusion matrices, ROC/PR curves

Each section includes conceptual explanations, practical code examples, real-world tips, common pitfalls, and interview Q&A focused on practical DS/ML engineering scenarios.