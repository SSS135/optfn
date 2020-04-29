from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, stacker_estimator, estimators):
        self.classifiers = estimators
        self.stacker_classifier = stacker_estimator

    def fit(self, x, y):
        predictions = [x]
        for classifier in self.classifiers:
            classifier.fit(x, y)
            pred_fn = classifier.predict_proba if hasattr(classifier, 'predict_proba') else classifier.predict
            predictions.append(pred_fn(x).reshape(x.shape[0], -1))
        self.stacker_classifier.fit(np.hstack(predictions), y)

    def predict_proba(self, x):
        predictions = [x]
        for classifier in self.classifiers:
            pred_fn = classifier.predict_proba if hasattr(classifier, 'predict_proba') else classifier.predict
            predictions.append(pred_fn(x).reshape(x.shape[0], -1))
        return self.stacker_classifier.predict_proba(np.hstack(predictions))