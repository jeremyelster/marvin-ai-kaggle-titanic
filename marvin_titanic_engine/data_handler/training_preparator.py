#!/usr/bin/env python
# coding=utf-8

"""TrainingPreparator engine action.

Use this module to add the project main code.
"""
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_score, GridSearchCV

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseDataHandler

__all__ = ['TrainingPreparator']


logger = get_logger('training_preparator')


class TrainingPreparator(EngineBaseDataHandler):

    def __init__(self, **kwargs):
        super(TrainingPreparator, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        """
        Setup the dataset with the transformed data that is compatible
        with the algorithm used to build the model in the next action.
        Use the self.initial_dataset prepared in the last action as source
        of data.

        Eg.

            self.marvin_dataset = {...}
        """
        pred_cols = ['Age', 'Pclass', 'Sex']
        dep_var = 'Survived'
        train_no_na = self.marvin_initial_dataset['train'][
            pred_cols + [dep_var]
        ].dropna()
        print("Length: {}".format(len(train_no_na)))

        # Feature Engineering
        data_X = train_no_na[pred_cols]
        data_X.loc[:, 'Sex'] = data_X['Sex'].map({'male': 1, 'female': 0})
        data_y = train_no_na[dep_var]

        # Prepare for Stratified Shuffle Split
        sss = StratifiedShuffleSplit(n_splits=5, test_size=.6, random_state=0)
        sss.get_n_splits(data_X, data_y)
        for train_index, test_index in sss.split(data_X, data_y):
            X_train, X_test = data_X.iloc[train_index], data_X.iloc[test_index]
            y_train, y_test = data_y.iloc[train_index], data_y.iloc[test_index]
        self.marvin_dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'sss': sss
        }
