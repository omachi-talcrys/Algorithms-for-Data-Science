# session8_decision_tree.py
# Practice script based on "Decision Tree Algorithm" session

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# -----------------------------
# 1. Simple Decision Tree Classifier (from scratch)
# -----------------------------
class SimpleDecisionTreeClassifier:
    def __init__(self):
        self.tree = None
    
    def gini(self, y):
        """Calculate the Gini Impurity for a list of labels."""
        prob = np.bincount(y) / len(y)
        return 1 - np.sum(prob ** 2)
    
    def best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        best_gini = float('inf')
        best_idx = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        for idx in range(n_features):
            thresholds = np.unique(X[:, idx])
            for threshold in thresholds:
                left_indices = X[:, idx] <= threshold
                right_indices = X[:, idx] > threshold
                if len(np.unique(y[left_indices])) == 0 or len(np.unique(y[right_indices])) == 0:
                    continue
                    
                left_gini = self.gini(y[left_indices])
                right_gini = self.gini(y[right_indices])
                weighted_gini = (len(y[left_indices]) * left_gini + len(y[right_indices]) * right_gini) / n_samples
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_idx = idx
                    best_threshold = threshold
        
        return best_idx, best_threshold
    
    def build_tree(self, X, y):
        """Build the decision tree recursively."""
        if len(np.unique(y)) == 1:
            return {'label': y[0]}
        
        best_idx, best_threshold = self.best_split(X, y)
        if best_idx is None:
            return {'label': np.bincount(y).argmax()}
        
        left_indices = X[:, best_idx] <= best_threshold
        right_indices = X[:, best_idx] > best_threshold
        return {
            'feature_index': best_idx,
            'threshold': best_threshold,
            'left': self.build_tree(X[left_indices], y[left_indices]),
            'right': self.build_tree(X[right_indices], y[right_indices])
        }

    def train(self, X, y):
        """Fit the decision tree on the data."""
        self.tree = self.build_tree(X, y)

    def predict_sample(self, node, x):
        """Predict a single sample based on the built tree."""
        if 'label' in node:
            return node['label']
        
        if x[node['feature_index']] <= node['threshold']:
            return self.predict_sample(node['left'], x)
        else:
            return self.predict_sample(node['right'], x)
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        return np.array([self.predict_sample(self.tree, x) for x in X])


# -----------------------------
# 2. Example Usage of Scratch Classifier
# -----------------------------
if __name__ == "__main__":
    # Example data
    X = np.array([[2, 3], [1, 2], [3, 6], [6, 7], [5, 8]])
    y = np.array([0, 0, 1, 1, 1])
    
    clf = SimpleDecisionTreeClassifier()
    clf.train(X, y)
    predictions = clf.predict(X)
    print("Scratch Decision Tree Predictions:", predictions)

    # -----------------------------
    # 3. Scikit-learn Decision Tree Classifier
    # -----------------------------
    X_cls = [[0, 0], [1, 1], [2, 2], [3, 3]]
    y_cls = [0, 1, 1, 0]
    
    classifier = DecisionTreeClassifier(max_depth=2)
    classifier.fit(X_cls, y_cls)
    print("Sklearn Classifier Prediction for [2,2]:", classifier.predict([[2, 2]]))
    
    plt.figure(figsize=(6, 4))
    plot_tree(classifier, filled=True, feature_names=["Feature1", "Feature2"], class_names=["Class0", "Class1"])
    plt.title("Decision Tree Classifier")
    plt.show()

    # -----------------------------
    # 4. Scikit-learn Decision Tree Regressor
    # -----------------------------
    X_reg = [[1], [2], [3], [4], [5]]
    y_reg = [2.3, 2.1, 3.8, 4.5, 5.0]
    
    regressor = DecisionTreeRegressor(max_depth=3)
    regressor.fit(X_reg, y_reg)
    prediction = regressor.predict([[3.5]])
    print("Sklearn Regressor Prediction for 3.5:", prediction)
    
    plt.figure(figsize=(6, 4))
    plot_tree(regressor, filled=True, feature_names=["Feature"], rounded=True)
    plt.title("Decision Tree Regressor")
    plt.show()
