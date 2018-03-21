#!/usr/bin/env python
# coding=utf-8

"""MetricsEvaluator engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from sklearn import metrics
import numpy as np

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['MetricsEvaluator']


logger = get_logger('metrics_evaluator')


class MetricsEvaluator(EngineBaseTraining):

    def __init__(self, **kwargs):
        super(MetricsEvaluator, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        """
        Setup the metrics with the result of the algorithms used to test the model.
        Use the self.dataset and self.model prepared in the last actions.

        Eg.

            self.marvin_metrics = {...}
        """


        for model_type, fitted_model in self.marvin_model.iteritems():
            print("Model Type: {0}\n{1}".format(
                model_type, fitted_model.best_estimator_.get_params())
            )
            print("Accuracy Score: {}%".format(
                round(fitted_model.best_score_, 4))
            )
            # Print the classification report of `y_test` and `predicted`
            print("Classification Report:\n")
            print(
                metrics.classification_report(
                    fitted_model.predict(self.marvin_dataset['X_test']),
                    self.marvin_dataset['y_test'])
            )

            # Print the confusion matrix
            print("Confusion Matrix:\n")
            print(
                metrics.confusion_matrix(
                    fitted_model.predict(self.marvin_dataset['X_test']),
                    self.marvin_dataset['y_test']
                )
            )
            print("\n\n")
            importances = self.marvin_model['rf'].best_estimator_.feature_importances_

            indices = np.argsort(importances)[::-1]

            # Print the feature ranking
            print("Feature ranking:")

            for f in range(self.marvin_dataset['X_train'].shape[1]):
                print("%d. feature %s (%f)" % (
                    f + 1,
                    self.marvin_dataset['X_train'].columns[indices[f]],
                    importances[indices[f]]
                ))

            self.marvin_metrics = {}
