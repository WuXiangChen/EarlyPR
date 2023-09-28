import numpy as np
import math

import pandas as pd


class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, predicted_class=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.predicted_class = predicted_class

class DecisionTreeClassifier:
    def __init__(self, max_depth=8, min_samples_split=15):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_classes = None
        self.n_features = None

    def _calculate_entropy(self, y):
        # Entropy(S) = -p_1 * log2(p_1) - p_2 * log2(p_2) - ... - p_k * log2(p_k)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        entropy = 0
        for count in class_counts:
            probability = count / total_samples
            entropy -= probability * np.log2(probability)
        return entropy

    def _calculate_information_gain(self, parent_entropy, left_y, right_y):
        # Information Gain(S, A) = Entropy(S) - Weighted_Entropy(S, A)
        total_samples = len(left_y) + len(right_y)
        left_weight = len(left_y) / total_samples
        right_weight = len(right_y) / total_samples
        left_entropy = self._calculate_entropy(left_y)
        right_entropy = self._calculate_entropy(right_y)

        weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
        information_gain = parent_entropy - weighted_entropy
        return information_gain


    def _calculate_split_information(self, left_y, right_y):
        # Split Information(S, A) = -p * log2(p) - q * log2(q)
        total_samples = len(left_y) + len(right_y)
        left_ratio = len(left_y) / total_samples
        right_ratio = len(right_y) / total_samples
        if left_ratio == 0 or right_ratio == 0:
            split_information = 0
        else:
            split_information = -left_ratio * math.log2(left_ratio) - right_ratio * math.log2(right_ratio)

        return split_information

    def _calculate_gain_ratio(self, parent_entropy, left_y, right_y):
        # Gain Ratio(S, A) = Gain(S, A) / SplitInformation(S, A)
        information_gain = self._calculate_information_gain(parent_entropy, left_y, right_y)
        split_information = self._calculate_split_information(left_y, right_y)
        if split_information == 0:
            gain_ratio = 0
        else:
            gain_ratio = information_gain / split_information
        return gain_ratio

    def _best_split(self, X, y):
        best_gain_ratio = 0
        best_split_feature = None
        best_split_value = None
        parent_entropy = self._calculate_entropy(y)
        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_indices = X[:, feature] <= value
                right_indices = X[:, feature] > value
                left_y = y[left_indices]
                right_y = y[right_indices]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                gain_ratio = self._calculate_gain_ratio(parent_entropy, left_y, right_y)
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_split_feature = feature
                    best_split_value = value
        #返回最佳分裂特征索引和最佳分裂阈值
        return best_split_feature, best_split_value

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = TreeNode(predicted_class=predicted_class)
        if depth < self.max_depth and len(y) >= self.min_samples_split:
            best_feature, best_value = self._best_split(X, y)
            if best_feature is not None:
                left_indices = X[:, best_feature] <= best_value
                right_indices = X[:, best_feature] > best_value
                left_X, left_y = X[left_indices], y[left_indices]
                right_X, right_y = X[right_indices], y[right_indices]
                node.feature_index = best_feature
                node.threshold = best_value
                node.left = self._grow_tree(left_X, left_y, depth + 1)
                node.right = self._grow_tree(right_X, right_y, depth + 1)
        return node

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X.values, y)

    def _predict(self, inputs, node):
        if node.left is None and node.right is None:
            return node.predicted_class
        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)

    def predict(self, X):
        re = np.array([self._predict(inputs, self.tree) for inputs in X])
        return re

    def predictall(self, X):
        X = X.values
        re = np.array([self._predict(inputs, self.tree) for inputs in X])
        return re

    def predict_lgbm(self, X):
        re = np.array([self._predict(inputs, self.tree) for inputs in X])
        return re

DecisionTree_clf = DecisionTreeClassifier(max_depth=8, min_samples_split=15)
DecisionTreeMerged_clf = DecisionTreeClassifier(max_depth=8, min_samples_split=15)



# from sklearn.tree import DecisionTreeClassifier
# DecisionTree_clf = DecisionTreeClassifier()

