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
import pickle
import heapq
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler


class Tokenizer(object):

    def __init__(self, code_len: int, start_id: int):
        self.code_len = code_len
        self.start_id = start_id
        self.end_id = start_id

    def encode(self, item: str) -> List[int]:
        raise NotImplementedError()

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError()

    @property
    def code_range(self) -> List[Tuple[int, int]]:
        """
        get the vocab id range for each of the encoded tokens
        @returns [(min, max), (min, max), ...]
        """
        return [(self.start_id, self.end_id)]


class FloatTokenizer(Tokenizer):
    """
    A quantization tokenization applied to float columns
    A float column is quantized by the following procedure:
    - set min = 0.0 by subtracting by float_colmin()
    - Compress these real numbers by taking log1p
    - compute the max digits of these logarithmed values using log10
    - allow for extra precision using the `extra_digits`
    - compute and save a lookup and reverse-lookup table for the digits up the precision defined by `extra_digits`

    so if the extra_digits is 4 and a logarithmed value is: 1.23456, we would truncate after the 4

    Computing the reverse transform follows the reverse steps, namely compute

    v = np.exp(v / 10**self.extra_digits) + min_value - 1.0
    """

    def __init__(self, col_name: str, data_series: DataFrame, start_id: int, transform='yeo-johnson'):

        self.name = col_name
        if transform == 'yeo-johnson':
            self.scaler = PowerTransformer(standardize=True)
        elif transform == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='uniform')
        elif transform == 'robust':
            self.scaler = RobustScaler()
        elif transform == 'log':
            pass
        else:
            raise ValueError('Supported data transformations are "yeo-johnson", "quantile", and "robust", and "log"')

        self.scaler_name = transform

        if not isinstance(data_series, DataFrame):
            raise ValueError(
                'pass in data_series as a dataframe using double bracket notation -> df[[col]] instead of df[col]')
        if self.scaler_name in ['yeo-johnson', 'quantile', 'robust']:
            # take unique values and enforce floats
            values = np.unique(self.scaler.fit_transform(data_series.values)).astype(float)
        else:
            values = np.unique(data_series.values).astype(float)

        # assume base 10 numbers, can change the base of the values if
        # larger dictionary is needed
        
        integral_digits = 0
        # digits to left of decimal
        integral_digits = int(np.log10(np.abs(values[~np.isnan(values)]).max()) + 1)
        # print(integral_digits)

        nans = True if any(np.isnan(values)) else False
        negatives = True if any(values < 0) else False
        positives = True if any(values >= 0) else False
        integral_digits += 1  # to handle the negative sign, nan, and + sign
        # print(integral_digits)
        decimal_digits = 0
        # digits to right of decimal - check if there are any values less than 1
        if np.abs(values[~np.isnan(values)]).min() < 1:
            # choose an extra sig fig for smallest value ex. 0.00013456 -> 0.00013
            # choose 2 smallest values for sig fig determination. If the smallest value == 0, then use second smallest
            sigfig = heapq.nsmallest(2, np.abs(values[~np.isnan(values)]))
            sigfig = sigfig[0] if sigfig[0] else sigfig[1]
            decimal_digits = int(abs(np.floor(np.log10(sigfig))) + 1)

        self.code_len: int = integral_digits + decimal_digits
        self.integral_digits = integral_digits

        self.decimal_digits = decimal_digits
        # super is here because we had to calculate code_len
        super().__init__(self.code_len, start_id)

        self.digits_id_to_item = [{} for _ in range(self.code_len)]
        self.digits_item_to_id = [{} for _ in range(self.code_len)]
        if self.scaler_name in ['yeo-johnson', 'quantile', 'robust']:
            self.scaler_vocab(positives, negatives, nans)
        else:
            self.log_vocab(values)

    def scaler_vocab(self, positives, negatives, nans):
        for i in range(self.code_len):
            if not i:
                #  parity and nans. Some data may be exclusively negative/positive
                if negatives:
                    self.add_to_vocab(i, '-')
                if nans:
                    self.add_to_vocab(i, 'nan')
                if positives:
                    # for positives
                    self.add_to_vocab(i, '+')
                continue
            # allow all digits to be accessible, ex. tens, hundredths, etc., can be adjusted to only allow data's digits
            for digit in range(10):
                self.add_to_vocab(i, str(digit))
            if nans:
                # placeholder for nans for remaining digits, or if the scaler produces integral values like 0 or 1
                # todo alternatively, if integral digits, add 0's instead
                self.add_to_vocab(i, '')

    def log_vocab(self, data_series):
        """original way of float tokenization"""
        self.mval = data_series.min()
        code_len = self.code_len
        # values are larger than zero
        # use log transformation to reduce the gap
        values = np.log(data_series - self.mval + 1.0)
        # assume base 10 numbers, can change the base of the values if
        # larger dictionary is needed
        digits = int(np.log10(values.max())) + 1
        extra_digits = code_len - digits
        if extra_digits < 0:
            raise "need large length to code the nummber"
        significant_val = (values * 10 ** extra_digits).astype(int)

        # adjust code len if not sized appropriately
        self.code_len = len(str(significant_val.max())) if len(str(significant_val.max())) > self.code_len else self.code_len

        self.extra_digits = extra_digits
        digits_id_to_item = [{} for _ in range(code_len)]
        digits_item_to_id = [{} for _ in range(code_len)]
        for i in range(code_len):
            id_to_item = digits_id_to_item[i]
            item_to_id = digits_item_to_id[i]
            v = (significant_val // 10 ** i) % 10
            uniq_items = np.unique(v)
            for i in range(len(uniq_items)):
                item = str(uniq_items[i])
                item_to_id[item] = self.end_id
                id_to_item[self.end_id] = item
                self.end_id += 1
        self.digits_id_to_item = digits_id_to_item
        self.digits_item_to_id = digits_item_to_id

    def add_to_vocab(self, i, item):
        """
        helper method
        i: the index of the dictionary for digits_id_to_item and digits_item_to_id
        item: the hashable object to be added to both dictionaries
        """
        self.digits_id_to_item[i][self.end_id] = item
        self.digits_item_to_id[i][item] = self.end_id
        self.end_id += 1

    @property
    def code_range(self) -> List[Tuple[int, int]]:
        """
        get the vocab id range for each of the encoded tokens
        @returns [(min, max), (min, max), ...]
        """
        # first largest digits
        outputs = []
        for i in range(self.code_len):
            ids = self.digits_id_to_item[i].keys()
            outputs.append((min(ids), max(ids) + 1))
        return outputs

    def encode(self, item: str) -> List[int]:
        if self.scaler_name == 'log':
            # old method, does not handle nans
            if item == 'nan':
                valstr = str(np.iinfo(np.int32).max)[:self.code_len]
                # double check nan digits are in dictionary
                for i in reversed(range(self.code_len)):
                    if valstr[self.code_len - i - 1] not in self.digits_item_to_id[i]:
                        self.digits_item_to_id[i][valstr[self.code_len - i - 1]] = self.end_id
                        self.digits_id_to_item[i][self.end_id] = valstr[self.code_len - i - 1]
                        self.end_id += 1
            else:
                val = float(item)
                values = np.log(val - self.mval + 1.0)
                valstr = str(abs((values * 10 ** self.extra_digits).astype(int)))[:self.code_len]

            # adjust code len if not sized appropriately
            # raise ValueError("not right length")

            valstr = valstr.zfill(self.code_len)
            codes = []
            for i in reversed(range(self.code_len)):
                codes.append(self.digits_item_to_id[i][valstr[self.code_len - i - 1]])
            return codes

        try:
            value = self.scaler.transform([[float(item)]]).squeeze()
            # https://stackoverflow.com/questions/29849445/convert-scientific-notation-to-decimals
            value = ("%.17f" % value).rstrip('0')
        except ValueError as exc:
            raise exc

        # check for nan and negative
        if value == 'nan':  # np.isnan(value):
            # encoding is easy, just use the tokens we know about already
            return [self.digits_item_to_id[i]['nan'] if not i else self.digits_item_to_id[i]['']
                    for i in range(self.code_len)]

        integer, decimal = str(value).split('.')
        missing_integer_digits = self.integral_digits - len(integer)
        missing_decimal_digits = self.decimal_digits - len(decimal)

        ids = []
        if integer[0] == '-':
            # check if ['0'] is good or use [''] and comment out if nans: conditional above
            chrs = ['-'] + ['0'] * missing_integer_digits + list(integer)[1:] + list(decimal[:self.decimal_digits]) + \
                   ['0'] * missing_decimal_digits
        else:
            chrs = ['+'] + ['0'] * (missing_integer_digits-1) + list(integer) + list(decimal[:self.decimal_digits]) + \
                   ['0'] * missing_decimal_digits
        for idx, _chr in enumerate(chrs):
            ids.append(self.digits_item_to_id[idx][_chr])
        return ids

    def decode(self, ids: List[int]) -> str:
        if self.scaler_name == 'log':
            # old method
            items = []
            for i in reversed(range(self.code_len)):
                items.append(self.digits_id_to_item[i][ids[self.code_len - i - 1]])
            v = int("".join(items))
            v = v / 10**self.extra_digits
            v = np.exp(v) + self.mval - 1.0
            return "{:.5f}".format(v)

        items = []
        isnan = False
        for idx, i in enumerate(ids):
            if idx == self.integral_digits and items and not items[0] == 'nan':
                items.append('.')
            items.append(self.digits_id_to_item[idx][i])
            if not idx and items[0] == 'nan':
                isnan = True
        if isnan:
            isnan = False
            return 'nan'
        val = float(''.join(items))
        return str(self.scaler.inverse_transform([[val]]).squeeze().astype(str))


class CategoricalTokenizer(Tokenizer):

    def __init__(self, col_name: str, data_series: Series, start_id: int):
        super().__init__(1, start_id)
        self.name = col_name
        uniq_items = pd.Series(data_series.unique().astype(str))
        uniq_items.index += self.end_id
        self.id_to_item = uniq_items.to_dict()
        self.item_to_id = pd.Series(data=uniq_items.index, index=uniq_items.values).to_dict()
        self.end_id += len(uniq_items)

    def encode(self, item) -> List[int]:
        return [self.item_to_id[item]]

    def decode(self, ids: List[int]) -> str:
        return self.id_to_item[ids[0]]


class ColumnTokenizer:

    def __init__(self):
        self.column_tokenizers: Dict[str, Tokenizer] = {}
        self.columns = []
        self.sizes = []

    @property
    def vocab_size(self):
        return self.column_tokenizers[self.columns[-1]].end_id

    def register(self, name: str, ccode: Tokenizer):
        self.columns.append(name)
        self.column_tokenizers[name] = ccode
        self.sizes.append(ccode.code_len)

    def encode(self, col: str, item: str) -> List[int]:
        if col in self.column_tokenizers:
            return self.column_tokenizers[col].encode(item)
        else:
            raise ValueError("cannot encode")

    def decode(self, col: str, ids: List[int]) -> str:
        if col in self.column_tokenizers:
            return self.column_tokenizers[col].decode(ids)
        else:
            raise ValueError("cannot decode")

    def get_range(self, column_id: int) -> List[Tuple[int, int]]:
        return self.column_tokenizers[self.columns[column_id]].code_range
