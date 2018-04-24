#!/usr/bin/env python
# coding=utf-8

try:
    import mock

except ImportError:
    import unittest.mock as mock

from marvin_titanic_engine.data_handler import AcquisitorAndCleaner
import pandas as pd


@mock.patch('marvin_titanic_engine.data_handler.acquisitor_and_cleaner.pd.read_csv')
@mock.patch('marvin_titanic_engine.data_handler.acquisitor_and_cleaner.MarvinData.download_file')

def test_execute(download_file_mocked, read_csv_mocked, mocked_params):

    read_csv_mocked.return_value = pd.DataFrame(data={'Age': {0: 32, 1: 34, 2: 70, 3: 48, 4: 71}, 'Fare': {0: 55, 1: 495, 2: 843, 3: 544, 4: 675}, 'Pclass': {0: 2, 1: 1, 2: 1, 3: 3, 4: 1}, 'Sex': {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}, 'Survived': {0: 1, 1: 0, 2: 0, 3: 1, 4: 0}})

    ac = AcquisitorAndCleaner()
    ac.execute(params=mocked_params)

    download_file_mocked.assert_any_call("https://s3.amazonaws.com/marvin-engines-data/titanic/test.csv")
    download_file_mocked.assert_any_call("https://s3.amazonaws.com/marvin-engines-data/titanic/train.csv")


    assert str(ac.marvin_initial_dataset['train']['Survived'][0]) == '1'
    assert read_csv_mocked.call_count == 2
    assert download_file_mocked.call_count == 2

    assert not ac._params
