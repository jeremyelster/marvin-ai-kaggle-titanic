#!/usr/bin/env python
# coding=utf-8

"""Predictor engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBasePrediction

__all__ = ['Predictor']


logger = get_logger('predictor')


class Predictor(EngineBasePrediction):

    def __init__(self, **kwargs):
        super(Predictor, self).__init__(**kwargs)

    def execute(self, input_message, params, **kwargs):
        """
        Return the predicted value in a json parsable object format.
        Use the self.model and self.metrics objects if necessary.
        """
        input_message = [[50, 3, 0]]

        final_result = {
            "prediction1": self.marvin_model['rf'].predict(input_message)[0],
            "prediction2": self.marvin_model['svm'].predict(input_message)[0]

        }

        print("Final Result:\n{}".format(final_result))
