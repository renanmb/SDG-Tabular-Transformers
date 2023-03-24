from coder.tabular_tokenizer import TabularTokenizer
import json


if __name__ == "__main__":
    tokenizer = TabularTokenizer('credit_card_coder.pickle',
                                 special_tokens=['\n', '<|endoftext|>'],
                                 delimiter='|')

    # save the encoder and decoder
    ids = tokenizer.code_column.column_tokenizers['Amount'].encode('-499.0')
    print(ids)
    t = tokenizer.code_column.column_tokenizers['Amount'].decode(ids)
    print(t)

    for i in range(len(tokenizer.code_column.columns)):
        print(tokenizer.code_column.columns[i], tokenizer.code_column.get_range(i))

    with open('credit_card.jn', 'r') as f:
        for line in f:
            break

    text = json.loads(line)['text']
    r = tokenizer.tokenize_str(text)
    ids = tokenizer.convert_tokens_to_ids(r)
    print(ids)
    tex = tokenizer.convert_ids_to_tokens(ids)
    print(tex)
    print(r)
    print(tokenizer.decode(ids))

