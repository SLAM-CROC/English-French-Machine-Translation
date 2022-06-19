import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Lang import Lang
import preprocessingUtil
import random


def preprocessing(file_path, nrows):
    df = pd.read_csv(file_path, nrows=nrows)
    lines = df.values.tolist()
    print("The number of English-French sentences pairs:", len(lines))
    pairs = [[preprocessingUtil.normalizeString(str(s)) for s in l] for l in lines]
    MAX_LENGTH = 100
    pairs = preprocessingUtil.filterPairs(pairs, MAX_LENGTH)
    lang1 = df.columns.tolist()[0]
    lang2 = df.columns.tolist()[1]

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print(random.choice(pairs))  # random choice of pairs

    return input_lang, output_lang, pairs


if __name__ == "__main__":

    # full dataset not taking load
    df = pd.read_csv("./en-fr.csv", nrows=800000)

    lines = df.values.tolist()

    print("The number of English-French sentences pairs:", len(lines))

    pairs = [[preprocessingUtil.normalizeString(str(s)) for s in l] for l in lines]

    list1, list2 = preprocessingUtil.sentWordNum(pairs)

    print("Average number of words in English sentences:", np.average(list1))
    print("Average number of words in French sentences:", np.average(list2))

    print("Max number of words in English sentences:", max(list1), "; Min number of words in English sentences:", min(list1))
    print("Max number of words in French sentences:", max(list2), "; Min number of words in French sentences:", min(list2))

    flag = input("Show data distribution?[y/n]:")
    if flag == 'y':
        plt.hist(list1, bins=500)
        plt.show()
        plt.hist(list2, bins=500)
        plt.show()
    else:
        pass

    MAX_LENGTH = 100
    pairs = preprocessingUtil.filterPairs(pairs, MAX_LENGTH)

    print("The number of English-French sentences pairs after filtering:", len(pairs))

    lang1 = df.columns.tolist()[0]
    lang2 = df.columns.tolist()[1]

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Vocabulary volume:", input_lang.name, input_lang.n_words)
    print("Vocabulary volume:", output_lang.name, output_lang.n_words)

    print("\n Language vocabulary word2index dict:")
    print(input_lang.name, preprocessingUtil.dict_slice(input_lang.word2index, 0, 10))
    print(output_lang.name, preprocessingUtil.dict_slice(output_lang.word2index, 0, 10))

    print("\n Language vocabulary index2word dict:")
    print(input_lang.name, preprocessingUtil.dict_slice(input_lang.index2word, 0, 10))
    print(output_lang.name, preprocessingUtil.dict_slice(output_lang.index2word, 0, 10))

    print("\n Language vocabulary word2count dict:")
    print(input_lang.name, preprocessingUtil.dict_slice(input_lang.word2count, 0, 10))
    print(output_lang.name, preprocessingUtil.dict_slice(output_lang.word2count, 0, 10))

    print("\n Data sample (a sentences pair):")
    print(random.choice(pairs))  # random choice of pairs
