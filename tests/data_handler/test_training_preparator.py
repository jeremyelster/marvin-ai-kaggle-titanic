#!/usr/bin/env python
# coding=utf-8

try:
    import mock

except ImportError:
    import unittest.mock as mock

from marvin_titanic_engine.data_handler import TrainingPreparator


class TestTrainingPreparator:

    def test_execute(self, mocked_params):

        import pandas as pd
        test_dataset = {}
        test_dataset["train"] = pd.DataFrame(data={
            'Age': [32, 34, 70, 48, 71],
            'Fare': [55, 495, 843, 544, 675],
            'Pclass': [2, 1, 1, 3, 1],
            'Sex': ['female', 'male', 'female', 'male', 'male'],
            'Survived': [1, 0, 0, 1, 0],
            'GARBAGE': [1, 2, 3, 4, 5]
        }, index=[0, 1, 2, 3, 4])

        test_dataset["test"] = pd.DataFrame(data={
            'Age': [32, 34, 70, 48, 71],
            'Fare': [55, 495, 843, 544, 675],
            'Pclass': [2, 1, 1, 3, 1],
            'Sex': ['female', 'male', 'female', 'male', 'male'],
            'Survived': [1, 0, 0, 1, 0],
            'GARBAGE': [1, 2, 3, 4, 5]
        }, index=[0, 1, 2, 3, 4])

        mocked_params = {
            "pred_cols": ["Age", "Pclass", "Sex", "Fare"],
            "dep_var": "Survived"
        }

        ac = TrainingPreparator(initial_dataset=test_dataset)
        ac.execute(params=mocked_params)
        assert set(ac.marvin_dataset["X_train"].columns) == set(mocked_params["pred_cols"])
        # assert ac.marvin_dataset["y_train"] == pd.Series([1, 0, 0, 1, 0])
        assert not ac._params
