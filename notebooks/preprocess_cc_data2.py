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

ALL_MCCS = sorted([int(mcc.mcc) for mcc in iso18245.get_all_mccs()])

datafp = './data/card_transaction_fixed.pq'
gdf = cudf.read_parquet(datafp)
gdf['zip'] = gdf.zip.astype(int).astype(str)
gdf['zip'] = gdf['zip'].fillna('')

gdf['use chip'] = gdf['use chip'].str.strip()
gdf['merchant city'] = gdf['merchant city'].str.strip()

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

unq_errors = [i for i in 
                 set( 
                    open_sublist(
                        [i.split(',') for i in gdf.errors.value_counts().index.to_arrow().to_pylist()
                        ])
                 ) if i]

# split errors column to be one hot encoded
exploded = gdf['errors'].str.strip(',').str.split(',').explode()
raw_one_hot = cudf.get_dummies(exploded, columns=["errors"])
errs = raw_one_hot.groupby(raw_one_hot.index).sum()
gdf = cudf.concat([gdf, errs], axis=1)
gdf = gdf.rename(columns={col:f'errors_{col}' for col in unq_errors})


# add day of the week feature
gdf['dayofweek'] = cudf.to_datetime(gdf.date).dt.dayofweek

# we'll make the (bad) assumption that the users have all their credit cards at the very beginning of the
# simulation

num_cards_per_user = gdf[['user', 'card']].groupby('user').card.nunique()

num_cards_per_user = num_cards_per_user.rename('num_cards_per_user')

gdf = cudf.merge(gdf, num_cards_per_user, on='user', how='left')

del exploded, errs, raw_one_hot, num_cards_per_user
gc.collect()

# sort these values just in case!  gdf = gdf.sort_values('date').reset_index(drop=True)
gdf = gdf.sort_values('date').reset_index(drop=True)

gdf['use chip'], unq_use_chip = gdf['use chip'].factorize()


# frequency and agg stats...
grp_stats = gdf.groupby('card')['amount'].agg(['mean', 'std']).rename(columns={'mean': 'mean_card_amount', 
                                                                               'std': 'std_card_amount'})

gdf['mean_card_amount'] = gdf['card'].map(grp_stats.to_pandas().to_dict()['mean_card_amount'])
gdf['std_card_amount'] = gdf['card'].map(grp_stats.to_pandas().to_dict()['std_card_amount'])
gdf['card_counts'] = gdf['card'].map(gdf.card.value_counts().to_pandas().to_dict())

# remove nans in strings
gdf['zip'] = gdf.zip.astype(str)

# fill nans with empty string...
gdf['zip'] = gdf['zip'].fillna('')
gdf['merchant state'] = gdf['merchant state'].fillna('')


# yippy kai yay!
# gdf.isna().sum()

# aggregation
gdf['merchant_city_state_zip'] = gdf['merchant city'] + ' ' +\
                                 gdf['merchant state'] + ' ' +\
                                 gdf['zip']

# remove any extra trailing whitespace
gdf['merchant_city_state_zip'] = gdf['merchant_city_state_zip'].str.strip()
# factorize merchant names since min/max of their names (which is an int) is at the bounds of 64 bit int
gdf['merchant name'], merch_name_uniq = gdf['merchant name'].factorize()
gdf['merchant_city_state_zip'], merch_city_state_zip_uniq = gdf['merchant_city_state_zip'].factorize()
gdf['merchant city'], merc_city_uniq = gdf['merchant city'].factorize()
gdf['merchant state'], merc_state_uniq = gdf['merchant state'].factorize()
gdf['zip'], zip_uniq = gdf['zip'].factorize()

# range from 0 to 29 inclusive
# for inference, will need to add back 1991 years.
min_year = gdf.year.min()
gdf['year'] = gdf.year-gdf.year.min()

gdf['mcc'] = gdf['mcc'].map({ALL_MCCS[i]: i for i in range(len(ALL_MCCS))})

hrs_since_last_txn = gdf[['user', 'date']].to_pandas().groupby('user').diff().astype('timedelta64[s]')/3600

hrs_since_last_txn = hrs_since_last_txn.rename(columns={'date': 'hrs_since_last_txn'})

# time_since_last_txn.astype('timedelta64[s]')/3600

gdf['hrs_since_last_txn'] = cudf.concat([gdf, cudf.from_pandas(hrs_since_last_txn)], axis=1)[['user', 'hrs_since_last_txn']]\
                                .to_pandas().groupby('user').transform(lambda x: x.fillna(x.mean()))['hrs_since_last_txn']
del hrs_since_last_txn
assert gdf['hrs_since_last_txn'].min() >= 0, 'sort values by datetime before calculating hrs_since_last_txn'


print('saving mappings...')
mapping_fp = './data/mappings.pkl'
with open(mapping_fp, 'wb') as f:
    pickle.dump({'merch_mapping': {merch: idx for idx, merch in enumerate(merch_name_uniq.to_arrow().tolist())},
                 'merc_city_uniq': {city: idx for idx, city in enumerate(merc_city_uniq.to_arrow().tolist())},
                 'merc_state_uniq': {state: idx for idx, state in enumerate(merc_state_uniq.to_arrow().tolist())},
                 'unq_use_chip': {i: idx for idx, i in enumerate(unq_use_chip.values_host)},
                 'zip_uniq': {zip_uniq[i]: i for i in range(len(zip_uniq))},
                 'min_year': min_year,
                }, f)

out_datafp = './data/card_transaction_processed.parquet'
gdf.to_parquet(out_datafp)
print('process complete on real data...')
