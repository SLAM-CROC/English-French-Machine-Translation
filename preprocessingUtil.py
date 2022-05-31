import unicodedata
import re


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def sentWordNum(en_fr_sent_pairs):
    enSentWordNumList = []
    frSentWordNumList = []
    for pair in en_fr_sent_pairs:
        enSentWordNumList.append(len(pair[0].split(' ')))
        frSentWordNumList.append(len(pair[1].split(' ')))
    return enSentWordNumList, frSentWordNumList


def filterPair(p, MAX_LENGTH):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs, MAX_LENGTH):
    return [pair for pair in pairs if filterPair(pair, MAX_LENGTH)]


def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice
