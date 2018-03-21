#!/usr/bin/env python
# coding=utf-8

"""Trainer engine action.

Use this module to add the project main code.
"""

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler, scale
# from sklearn.linear_model import LogisticRegression

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['Trainer']


logger = get_logger('trainer')


class Trainer(EngineBaseTraining):

    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        """
        Setup the model with the result of the algorithm used to training.
        Use the self.dataset prepared in the last action as source of data.

        Eg.

            self.marvin_model = {...}
        """

        # Set the parameter candidates
        parameter_candidates = [
            {'C': [1, 10, 100], 'gamma': [0.01, 0.001], 'kernel': ['linear']},
            {'C': [1, 10, 100], 'gamma': [0.01, 0.001], 'kernel': ['rbf']},
        ]

        # Create a classifier with the parameter candidates
        svm_grid = GridSearchCV(
            estimator=svm.SVC(),
            param_grid=parameter_candidates,
            n_jobs=-1
        )

        # Train the classifier on training data
        svm_grid.fit(
            self.marvin_dataset['X_train'],
            self.marvin_dataset['y_train']
        )

        # use a full grid over all parameters
        parameter_candidates = {
            "max_depth": [3, None],
            "random_state": [0],
            "min_samples_split": [2],  # , 3, 10],
            "min_samples_leaf": [1],  # , 3, 10],
            "n_estimators": [20],  # , 50],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"]
        }

        # run grid search
        rf_grid = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=parameter_candidates
        )
        rf_grid.fit(
            self.marvin_dataset['X_train'],
            self.marvin_dataset['y_train']
        )

        self.marvin_model = {
            'svm': svm_grid,
            'rf': rf_grid
        }
