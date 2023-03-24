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
from typing import Iterable, List, Optional

import numpy as np

from .column_code import ColumnTokenizer


class TabularTokenizer(object):

    def __init__(self, coder_file,
                 special_tokens=None,
                 delimiter=','):
        if special_tokens is None:
            special_tokens = ['<|endoftext|>', '\n']
        elif '<|endoftext|>' not in special_tokens:
            raise ValueError('<|endoftext|> token must be present as a special token')

        if delimiter in special_tokens:
            raise ValueError('delimiter can not be one of the special tokens')

        try:
            with open(coder_file, 'rb') as handle:
                self.code_column: ColumnTokenizer = pickle.load(handle)
        except FileNotFoundError as exc:
            print(coder_file)
            raise exc
        except Exception as exc:
            print(coder_file)
            raise exc
        self.num_columns = len(self.code_column.columns)
        self.special_tokens = {}
        self.special_tokens_decoder = {}
        self.set_special_tokens(special_tokens)
        self.delimiter = delimiter
        self.eod_id = self.special_tokens['<|endoftext|>']

    def __len__(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return max(self.special_tokens_decoder.keys()) + 1

    def tokenize(self, text):
        return self.encode(text)

    def detokenize(self, token_ids):
        return self.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def eor(self):
        return self.special_tokens['\n']

    def set_special_tokens(self, special_tokens: Iterable[Optional[str]]):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last
            index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            special_tokens = ['<|endoftext|>']
            # self.special_tokens = {}
            # self.special_tokens_decoder = {}
            # self.eod_id = None
            # return
        self.special_tokens = dict((tok, self.code_column.vocab_size + i)
                                   for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {
            v: k for k, v in self.special_tokens.items()}

        if '<|endoftext|>' in self.special_tokens:
            self.eod_id = self.special_tokens['<|endoftext|>']

    def tokenize_str(self, text: str) -> List[str]:
        """ Tokenize a string. """
        endoftext = '<|endoftext|>'
        tokens = []
        for doc in text.split(endoftext):  # handle case where endoftext token is in middle of string (2+ documents)
            rows = doc.split('\n')

            # num_rows = len(rows)
            used_eot = False
            for row in rows:
                if not row:
                    continue
                if not used_eot:
                    tokens.extend([endoftext] + ''.join(row.split(endoftext)).split(self.delimiter))  # split into fields
                    used_eot = True
                else:
                    tokens.extend(''.join(row.split(endoftext)).split(self.delimiter))  # split into fields
                tokens.append('\n')
        return tokens

    def convert_tokens_to_ids(self, tokens: List[str]):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        cindex = 0
        for token in tokens:

            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                index = cindex % self.num_columns
                column = self.code_column.columns[index]
                # print(len(ids), column)  # good to check if you get a keyerror
                ids.extend(self.code_column.encode(column, token))
                cindex += 1
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False) -> List:
        """Converts a sequence of ids in BPE tokens using the vocab."""
        tokens = []
        cindex = 0
        sizes = self.code_column.sizes
        ids_size = sum(sizes)
        cum_sizes = np.cumsum(sizes)
        old_column_index = -1
        token_ids = []
        keyerror = False
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens and not (tokens and bool(tokens[-1] in self.special_tokens)):
                    tokens.append(self.special_tokens_decoder[i])
            else:
                index = cindex % ids_size
                
                # keep iterating through the rest of the row
                if keyerror and index != 0:
                    cindex += 1
                    continue
                elif keyerror and index==0:
                    keyerror = False
                    
                column_index = np.where(index < cum_sizes)[0][0]
                column = self.code_column.columns[column_index]
                #  print(column, i)
                if old_column_index != column_index:
                    token_ids = [i]
                    old_column_index = column_index
                else:
                    token_ids.append(i)
                if not keyerror and (len(token_ids) == sizes[column_index]):
                    try:
                        tokens.append(self.code_column.decode(column, token_ids))
                    except KeyError as e:
                        keyerror = True
                        # remove the row, index is the number of tokens added, including the current one. Since the current column tokens aren't
                        # added we add back those so we don't remove more than a row
                        tokens = tokens[:-1 - index + len(token_ids)]
                        # if keyerror occurs on the first col, then iterate to col 1 and discard
                
                cindex += 1
        return tokens

    def encode(self, text: str) -> List:
        if isinstance(text, str):
            return self.convert_tokens_to_ids(self.tokenize_str(text))
        raise TypeError('text input to encode method should be a string')

    def decode(self, token_ids: List) -> str:
        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
        all_lines = []
        line = []
        for token in tokens:
            if token == '<|endoftext|>' or token == '\n':
                if len(line) != 0:
                    line_text = self.delimiter.join(line)
                    all_lines.append(line_text)
                all_lines.append(token)
                line = []
            else:
                line.append(token)
        #print(line)
        if len(line) != 0:
            # remaining items
            line_text = self.delimiter.join(line)
            all_lines.append(line_text)
        text = "".join(all_lines)
        #print(text)
        return text
