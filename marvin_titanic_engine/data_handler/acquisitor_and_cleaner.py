#!/usr/bin/env python
# coding=utf-8

"""AcquisitorAndCleaner engine action.

Use this module to add the project main code.
"""

import pandas as pd

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseDataHandler

__all__ = ['AcquisitorAndCleaner']


logger = get_logger('acquisitor_and_cleaner')


class AcquisitorAndCleaner(EngineBaseDataHandler):

    def __init__(self, **kwargs):
        super(AcquisitorAndCleaner, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        """
        Setup the initial_dataset with all cleaned data necessary to build your dataset in the next action.

        Eg.

            self.marvin_initial_dataset = {...}
        """
        train_df = pd.read_csv(
            'marvin_titanic_engine/data_files/train.csv'
        )
        test_df = pd.read_csv(
            'marvin_titanic_engine/data_files/train.csv'
        )
        self.marvin_initial_dataset = {
            'train': train_df,
            'test': test_df
        }
