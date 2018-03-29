#!/usr/bin/env python
# coding=utf-8

"""PredictionPreparator engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBasePrediction

__all__ = ['PredictionPreparator']


logger = get_logger('prediction_preparator')


class PredictionPreparator(EngineBasePrediction):

    def __init__(self, **kwargs):
        super(PredictionPreparator, self).__init__(**kwargs)

    def execute(self, input_message, params, **kwargs):
        # put this values in engine.messages to be used as dryrun samples
        # age, class, sex
        input_message = {"age": 50, "class": 3, "sex": 0}

        key_order = {"age": 0, "class": 1, "sex": 2}
        input_message = [input_message[i] for i in sorted(input_message, key=key_order.__getitem__)]

        return input_message
