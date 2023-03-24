import argparse
import os

import cudf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="params for creating dgl graph")
    parser.add_argument('-p', '--path', type=str, default='./data/card_transaction.v1.csv', help='raw file')
    parser.add_argument('-f', '--force', type=bool, default=False, help='force creation even if file exists')
    args = parser.parse_args()
    
    datafp = args.path #'./data/card_transaction.v1.csv'
    
    out_datafp = './data/card_transaction_fixed.pq'
    if args.force or not os.path.isfile(out_datafp):
    
        gdf = cudf.read_csv(datafp)

        # rename to lowercase
        gdf = gdf.rename(columns={'Errors?': 'errors',
                                  'Is Fraud?': 'is_fraud'
                                  })

        # split time
        gdf[['hour', 'minute']] = gdf.Time.str.split(':', expand=True)
        # rename to lowercase
        gdf.columns = [i.lower() for i in gdf.columns.tolist()]


        gdf.hour = gdf.hour.astype(int)
        gdf.minute = gdf.minute.astype(int)
        # add date col
        gdf['date'] = cudf.to_datetime(gdf[['year', 'month', 'day', 'hour', 'minute']])

        # sort rows by date and re-order the cols
        gdf = gdf.sort_values('date')\
                  [['user', 'card', 'date', 'year', 'month', 'day', 'time', 'hour', 'minute', 'amount',
                    'use chip', 'merchant name', 'merchant city', 'merchant state', 'zip',
                    'mcc', 'errors', 'is_fraud']]\
                 .reset_index(drop=True)

        # factorize is_fraud col into 1 and 0.
        gdf['is_fraud'], fraud_key = gdf['is_fraud'].factorize()

        # remove out the $ sign in the amount column and convert to float.
        gdf['amount'] = gdf.amount.str.strip('$').astype(float)
        gdf['zip'] = gdf['zip'].astype(int).astype(str)
        gdf['use chip'] = gdf['use chip'].str.strip()
        gdf['merchant city'] = gdf['merchant city'].str.strip()


        gdf.to_parquet(out_datafp)
    else:
        print('file exists already...')
    print('preprocessing complete...')
