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
import functools
import json
from multiprocessing import Pool, cpu_count
import os
import pickle
import time
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from coder.column_code import ColumnTokenizer, FloatTokenizer, CategoricalTokenizer


def get_docs_from_df(group) -> List:
    split_line = 500
    strike = 250
    group = group.sort_values(['year', 'month', 'day', 'hour', 'minute'])
    group = group.astype(str)  # convert to string for concatenation in loop below
    for col in ['hour', 'minute']:
        group[col] = group[col].str.zfill(2)
    result = group[["user"]].copy()
    result.columns = ["Out"]
    cols = group.columns[1:]
    total = []
    for col in cols:
        result["Out"] = result["Out"].str.cat(group[col], sep=delimiter)

    for start_rows in range(0, max(len(result) - split_line, len(result)), strike):
        if not start_rows:
            total.append(json.dumps({'text': eod_str + '\n'.join(
                result['Out'].iloc[start_rows: start_rows + split_line].to_json(orient='values')[2:-2]\
                             .replace('\\', '').split('","'))}) + '\n')
        else:
            total.append(json.dumps({'text': '\n'.join(
                result['Out'].iloc[start_rows: start_rows + split_line].to_json(orient='values')[2:-2]\
                             .replace('\\', '').split( '","'))}) + '\n')

    return total


def apply_parallel(dfGrouped, func, convert_to_series=False):
    """
    pandas Groupby apply func to groups in parallel
    https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby?noredirect=1&lq=1
    """

    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    if convert_to_series:
        return pd.concat(ret_list)
    return ret_list


def gen_one_doc(user_group, n_cores):
    udfs = list(user_group)
    pool = Pool(n_cores)
    docs = pool.map(get_docs_from_df, udfs)
    pool.close()
    pool.join()
    return functools.reduce(lambda a, b: a + b, docs)


def preprocess_raw_data(fp: str) -> pd.DataFrame:
    """
    preprocess the raw transaction data
    fp(str): path to the raw transaction data.
    """
    df = pd.read_csv(fp)
    df = df.rename(columns={'Errors?': 'errors',
                            'Is Fraud?': 'is_fraud'
                            })

    # split time into hour and minute
    df[['hour', 'minute']] = df.Time.str.split(':', expand=True)
    df.hour = df.hour.astype(int)
    df.minute = df.minute.astype(int)

    # remove the 'Time' Column once it is parsed
    del df['Time']
    # rename all cols to lowercase
    df.columns = [i.lower() for i in df.columns]
    # add date col
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])

    # remove the dollar string and convert amount to float
    df['amount'] = df['amount'].str.replace('$', '', regex=False).astype('float')

    # sort rows by date and re-order the cols
    df = df.sort_values('date')[['user', 'card', 'date', 'year', 'month', 'day', 'hour', 'minute', 'amount',
                                 'use chip', 'merchant name', 'merchant city', 'merchant state', 'zip',
                                 'mcc', 'errors', 'is_fraud']].reset_index(drop=True)

    # fix the zip column
    df['zip'] = df['zip'].apply(lambda x: '' if pd.isna(x) else "{:05.0f}".format(x))
    # factorize is_fraud col into 1 and 0.
    df['is_fraud'], fraud_key = df['is_fraud'].factorize()
    df['use chip'] = df['use chip'].str.strip()
    df['merchant city'] = df['merchant city'].str.strip()

    df.to_parquet(out_data_fp)
    print('Complete')
    return df


def tokenize_columns(df, float_cols: Optional[List[str]] = None, use_gpu: Optional[bool] = False):
    """
    Create a tokenizer for the columns in the DataFrame
    """
    beg = 0
    cc = None
    if not float_cols:
        float_cols = []

    num_float_tokens = 4
    if use_gpu:
        column_codes = ColumnTokenizer()
        for column in df.columns:
            start_id = beg if cc is None else cc.end_id
            if column in float_cols:
                cc = FloatTokenizer(column, df[column], num_float_tokens, start_id)
            else:
                cc = CategoricalTokenizer(column, df[column], start_id)
            column_codes.register(column, cc)
            print(f'{column}: ({start_id}, {cc.end_id})')
        print('Each row uses', sum(column_codes.sizes) + 1, 'tokens')

    else:
        column_codes = ColumnCodes()
        for column in df.columns:
            start_id = beg if cc is None else cc.end_id

            if column in float_cols:
                cc = FloatCode(column, df[column], num_float_tokens, start_id)
            else:
                cc = ColumnCode(column, df[column], start_id)
            column_codes.register(column, cc)
            print(f'{column}: ({start_id}, {cc.end_id})')
        print('Each row uses', sum(column_codes.sizes) + 1, 'tokens')
    return column_codes


if __name__ == "__main__":
    delimiter = '|'
    eod_str = '<|endoftext|>'
    split_line = 500
    strike = 250
    float_columns = ["amount"]

    # file paths
    data_fp = './data/card_transaction.v1.csv'
    out_data_fp = './data/card_transactions_fixed.parquet'
    pickle_fp = 'credit_card_coder.pickle'
    json_lines_fp = 'credit_card.jn'
    use_gpu = True

    if use_gpu:
        import cudf
        df = cudf.read_parquet(out_data_fp) if os.path.isfile(out_data_fp) else preprocess_raw_data(data_fp)
    else:
        df = pd.read_parquet(out_data_fp) if os.path.isfile(out_data_fp) else preprocess_raw_data(data_fp)
    df = df.drop(columns='date') if 'date' in df.columns else df

    column_codes = tokenize_columns(df, float_columns, use_gpu=use_gpu)
    # save the encoder and decoder
    with open(pickle_fp, 'wb') as handle:
        pickle.dump(column_codes, handle)

    df = df.to_pandas() if use_gpu else df

    with open(json_lines_fp, 'w') as f:
        # docs = gen_one_doc(user_group, 30)
        start = time.time()
        docs = apply_parallel(df.groupby('user'), get_docs_from_df)
        print(f"elapsed (s): {time.time() - start: 0.3f}")
        for doc in tqdm(docs):
            f.write(''.join(doc))
