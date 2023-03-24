import gc
import os
import pickle

import cudf
import cupy as cp
import iso18245
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm.autonotebook import tqdm

pd.set_option('display.max_columns', 500)

sns.set()

DEVICE = 0
cp.cuda.Device(DEVICE).use()


def open_sublist(array: list):
    """
    recursively opens up sublists
    """
    l = []
    for item in array:
        if isinstance(item, list):
            subl = open_sublist(item)
            l.extend(subl)
            continue
        l.append(item)
    return l


ALL_MCCS = sorted([int(mcc.mcc) for mcc in iso18245.get_all_mccs()])
with open('./data/mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

gen = cudf.read_csv('./data/txn_nov.csv')
gen.columns = [i.lower() if '?' not in i else i.lower()[:-1] for i in gen.columns]
gen = gen[gen.columns[1:]]
if 'is fraud' in gen.columns:
    gen = gen.rename(columns={'is fraud': 'is_fraud'})
gen['zip'] = gen.zip.astype(int).astype(str).str.zfill(5)

#dropping group and reordering cols
gen = gen[['user', 'card', 'year', 'month', 'day', 'hour', 'minute', 'amount',
           'use chip', 'merchant name', 'merchant city', 'merchant state', 'zip',
           'mcc', 'errors', 'is_fraud']]

print(f'Because we generate a new user if a user appears again in a group, there are {gen.loc[gen.user>=1999].user.nunique()} such cases resulting in {len(gen.loc[gen.user>=1999])} transactions\nMerging these users...')
gen.user = gen.user % 10000

assert gen.loc[gen.user>=2000].empty


if gen['amount'].dtype.kind == 'O':
    weird_amounts = gen['amount'].str.contains('..', regex=False).sum()
    if weird_amounts:
        print(f'weird amounts in the dataset ({weird_amounts}). Fixing...')
        gen['amount'] = gen['amount'].str.replace('..', '.', regex=False).astype(float)

if gen.is_fraud.dtype.kind == 'O':
    gen.is_fraud, _ = gen.is_fraud.factorize()
    
if gen['is_fraud'].any() >1 or gen['is_fraud'].any() < 0:
    print('weird fraud values encountered in generated dataset')
    gen = gen.loc[ (gen.is_fraud >=0) & (gen.is_fraud <=1) ]
    
month = gen.month.astype(str)
month.loc[month.str.len() == 1] = '0' + month.loc[month.str.len() == 1]

hour = gen.hour.astype(str)
hour.loc[hour.str.len() == 1] = '0' + hour.loc[hour.str.len() == 1]

day = gen.day.astype(str)
day.loc[day.str.len() == 1] = '0' + day.loc[day.str.len() == 1]

minute = gen.minute.astype(str)
minute.loc[day.str.len() == 1] = '0' + minute.loc[minute.str.len() == 1]

gen['date'] = cudf.to_datetime(gen.year.astype(str) + '-' +  \
                               month  + '-' + \
                               day + ' ' +\
                               hour + ':' +\
                               minute, format='%Y-%m-%d %H:%M')

gen['merchant name'] = gen['merchant name'].map(mappings['merch_mapping'])
gen['merchant city'] = gen['merchant city'].map(mappings['merc_city_uniq'])
gen['merchant state'] = gen['merchant state'].map(mappings['merc_state_uniq'])
gen['mcc'] = gen['mcc'].map({ALL_MCCS[i]: i for i in range(len(ALL_MCCS))})

# remove nans in strings
gen['zip'] = gen.zip.astype(int).astype(str)
# fill nans with empty string...
gen['zip'] = gen['zip'].fillna('')
# found out that generated txns have min value of -1 for Online transactions, so fillna(0) to capture this after the mapping.
gen['zip'] = gen['zip'].map(mappings['zip_uniq']).fillna(0)

# check that everything is mapped
assert gen.zip.isna().sum() == 0


# exploded = gen['errors'].str.strip(',').str.split(',').explode()

# raw_one_hot = cudf.get_dummies(exploded, columns=["errors"])
# errs = raw_one_hot.groupby(raw_one_hot.index).sum()

# gen = cudf.concat([gen, errs], axis=1)

# gen = gen.rename(columns={col:f'errors_{col}' for col in unq_errors})

# add day of the week feature
gen['dayofweek'] = cudf.to_datetime(gen.date).dt.dayofweek


# we'll make the (bad) assumption that the users have all their credit cards at the very beginning of the
# simulation

num_cards_per_user = gen[['user', 'card']].groupby('user').card.nunique()

num_cards_per_user = num_cards_per_user.rename('num_cards_per_user')

gen = cudf.merge(gen, num_cards_per_user, on='user', how='left')

# del exploded, errs, raw_one_hot, num_cards_per_user
del num_cards_per_user

gc.collect()

gen = gen.sort_values('date').reset_index(drop=True)

hrs_since_last_txn = gen[['user', 'date']].to_pandas().groupby('user').diff().astype('timedelta64[s]')/3600

hrs_since_last_txn = hrs_since_last_txn.rename(columns={'date': 'hrs_since_last_txn'})

# time_since_last_txn.astype('timedelta64[s]')/3600

gen['hrs_since_last_txn'] = cudf.concat([gen, cudf.from_pandas(hrs_since_last_txn)], axis=1)[['user', 'hrs_since_last_txn']]\
                                .to_pandas().groupby('user').transform(lambda x: x.fillna(x.mean()))['hrs_since_last_txn']


# sometimes a generated user may only have a single transaction. Setting hrs_since_last_txn to median
single_txn = gen['hrs_since_last_txn'].isna().sum()
if single_txn:
    print(f'There are some users with only a single transaction ({single_txn}). Fixing by setting to median ...')
gen['hrs_since_last_txn'] = gen['hrs_since_last_txn'].fillna(gen['hrs_since_last_txn'].median())
assert gen['hrs_since_last_txn'].isna().sum() == 0 


gen['year'] = gen['year'] - mappings['min_year']

gen['use chip'] = gen['use chip'].map(mappings['unq_use_chip'])

assert gen['hrs_since_last_txn'].min() >= 0, 'sort values by datetime before calculating hrs_since_last_txn'

out_datafp = './data/synthetic_txns.parquet'
gen.to_parquet(out_datafp)

print('process complete on synthetic data...')