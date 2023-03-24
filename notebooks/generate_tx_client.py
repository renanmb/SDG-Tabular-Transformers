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
import json
import pickle
import sys
import time
from typing import Dict, List

import requests
from tqdm import tqdm

from coder.column_code import ColumnTokenizer, FloatTokenizer, CategoricalTokenizer
# from coder.column_code import ColumnCodes


def request_data(data: Dict) -> List[str]:
    resp = requests.put('http://localhost:{}/generate'.format(port_num),
                        data=json.dumps(data), headers=headers)
    sentences = resp.json()['sentences']
    return sentences


def get_condition_text(sentences: List, history_rows: int):
    condition_text = ['\n'.join([ss for ss in s.split('\n')[-(history_rows + 1):]]) for s in sentences]
    return condition_text


def get_extra_text(sentences: List[str], history_rows: int):
    extra_text = ['\n'.join([ss for ss in s.split('\n')[history_rows:]]) for s in sentences]
    return extra_text


if __name__ == '__main__':
    port_num = int(sys.argv[1])  # example: 5000
    prefix_name = sys.argv[2]  # example: synthetic

    with open('credit_card_coder.pickle', 'rb') as handle:
        cc: ColumnTokenizer = pickle.load(handle)

    # batch_size = 32
    batch_size = 16
    num_of_rows = 50
    token_per_rows = sum(cc.sizes) + 1
    history_rows = 40
    num_of_blocks = 3000

    headers = {"Content-Type": "application/json"}

    files = []
    for i in range(batch_size):
        files.append(open("{}_{}.txt".format(prefix_name, i), 'w'))

    # generate the inital transactions
    data = {
        "sentences": [""] * batch_size,
        "tokens_to_generate": num_of_rows * token_per_rows,
        "temperate": 1.0,
        "add_BOS": True
    }

    sentences = request_data(data)

    for i in range(batch_size):
        s = sentences[i]
        files[i].write(s.replace('<|endoftext|>', '\n'))

    # generate the transactions conditioned on the previous ones
    pbar = tqdm(range(num_of_blocks))

    for block in pbar:
        pbar.set_description("block id: {}".format(block))
        condition_text = get_condition_text(sentences, history_rows)
        # print('conditional text')
        # for s in condition_text:
        #     print(s)

        data = {
            "sentences": condition_text,
            "tokens_to_generate": num_of_rows * token_per_rows,
            "temperate": 1.0,
            "add_BOS": False
        }

        try:
            sentences = request_data(data)
        except Exception as exc:
            print(block, exc, condition_text, '\n\n')
            time.sleep(3)
            continue

        extra_text = get_extra_text(sentences, history_rows)

        for i in range(batch_size):
            s = extra_text[i]
            files[i].write(s.replace('<|endoftext|>', '\n'))
            files[i].flush()

    for i in range(batch_size):
        files[i].close()
