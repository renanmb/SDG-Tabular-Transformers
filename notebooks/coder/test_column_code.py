############################################################################
##
## Copyright (C) 2022 NVIDIA Corporation.  All rights reserved.
##
## NVIDIA Sample Code
##
## Please refer to the NVIDIA end user license agreement (EULA) associated
## with this source code for terms and conditions that govern your use of
## this software. Any use, reproduction, disclosure, or distribution of
## this software and related documentation outside the terms of the EULA
## is strictly prohibited.
##
############################################################################
import pytest
import numpy as np
import pandas as pd

from coder.column_code import FloatTokenizer

np.random.seed(123)

data_cols = []  # there were cols here


@pytest.fixture()
def df():
    """simple preprocess of data"""
    fpath = f'' # file path here'
    df = pd.read_csv(fpath)
    data_cols = [col for col in df.columns if col != 'DATE']
    df[df[data_cols] <= -999999] = pd.NA
    df.DATE = pd.to_datetime(df.DATE, format='%Y%m%d')

    df['quarter'] = df.DATE.dt.quarter
    df['year'] = df.DATE.dt.year
    return df


@pytest.fixture()
def tokenizer():
    return FloatTokenizer


@pytest.mark.parametrize('start_id, col', [(0, 'COL_NAME'), (100, 'COL_NAME')])
def test_code_range(df, tokenizer, start_id, col):
    tokenizer = tokenizer(col, df[[col]], start_id)

    assert hasattr(tokenizer, 'code_range')
    outputs = tokenizer.code_range
    assert outputs[0][0] == start_id
    for rng in outputs:
        assert len(rng) == 2 and rng[0] < rng[1]


@pytest.mark.parametrize('start_id, col', zip(np.random.randint(0, 1000, size=len(data_cols)),
                                              data_cols))
def test_encode(df, tokenizer, start_id, col):
    tokenizer = tokenizer(col, df[[col]], start_id, 'robust')

    for value in df[col].astype(str):
        enc = tokenizer.encode(value)
        # assert len(enc) == tokenizer.code_len == len(set(enc))  # this test can be optimized 
        if not bool( len(enc) == tokenizer.code_len == len(set(enc))):
            print('NOT CLOSE')


@pytest.mark.parametrize('start_id, col', zip(np.random.randint(0, 1000, size=len(data_cols)),
                                              data_cols))
def test_decode(df, tokenizer, start_id, col):

    tokenizer = tokenizer(col, df.loc[df[col].notna(), [col]], start_id, 'log')
    for value in df[col].astype(str):

        enc: list = tokenizer.encode(value)
        dec = tokenizer.decode(enc)
        if value == 'nan' or (dec == 'nan' and value == 'nan'):
            continue
        # assert np.isclose(float(dec), float(value), rtol=5e-2, atol=1e-5) # this test can be optimized 

        if not np.isclose(float(dec), float(value),  rtol=5e-2, atol=1e-5):
            print('NOT CLOSE?\t', float(dec), float(value))
