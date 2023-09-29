from .DecisionTree_clf import DecisionTreeClassifier
import numpy as np

class LGBMClassifier:
    def __init__(self, num_leaves=10, learning_rate=0.1, n_estimators=2):
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.models = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def huber_loss(self,residuals, delta):
        loss = np.where(np.abs(residuals) <= delta, 0.5 * residuals ** 2, delta * np.abs(residuals) - 0.5 * delta ** 2)
        return loss

    def _boost_round(self, y, w, y_pred):
        residuals = y - y_pred
        gradient = residuals
        loss = self.huber_loss(gradient, 0.5)
        hessian = loss * (1 - loss)
        z = w * np.exp(-self.learning_rate * gradient) / (2 * np.sqrt(hessian+1))
        return z

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        w = np.ones(X.shape[0])
        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=4)
            model.fit(X, y)
            y_pred = model.predict_lgbm(X.values)
            w = self._boost_round(y,w,y_pred)
            chosen_indices = np.random.choice(len(X), size=len(X)//2, p=w / np.sum(w))
            X = X.loc[chosen_indices].reset_index(drop=True)
            y = y.loc[chosen_indices].reset_index(drop=True)
            w = w.loc[chosen_indices].reset_index(drop=True)
            self.models.append(model)

    def predict_proba(self, X):
        preds = np.zeros((X.shape[0], 2))
        for model in self.models:
            preds += model.predict(X)
        return preds / len(self.models)

    def predict(self, X):
        X = X.values
        probs = self.predict_proba(X)
        return np.array([int(np.max(probs, axis=1))])

    def predictall(self, X):
        X = X.values
        probs = self.predict_proba(X)
        return np.max(probs, axis=1).astype(np.int16)

LGBM_trainer = LGBMClassifier()
LGBM_trainer_mergedPR = LGBMClassifier()


# import lightgbm as lgb
# LGBM_trainer = lgb.LGBMClassifier(objective='binary', metric='Classification error rate', verbosity=-1)
# LGBM_trainer_mergedPR = lgb.LGBMClassifier(objective='binary', metric='Classification error rate', verbosity=-1)
