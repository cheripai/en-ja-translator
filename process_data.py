import constants as c
import pickle
import re
from JapaneseTokenizer import JumanppWrapper
from Lang import Lang


def normalize_en(s):
    """ Processes an English string by removing non-alphabetical characters (besides .!?).
    """
    s = s.lower().strip()
    s = re.sub(r"[^\w.!?]+", r" ", s, flags=re.UNICODE)
    return s


def normalize_ja(s, segmenter):
    """ Processes a Japanese character by removing non-word characters and separating tokens with spaces.
    """
    s = re.sub(r"[^\w.!?。]+", r" ", s, flags=re.UNICODE)
    s = " ".join(segmenter.tokenize(sentence=s, is_feature=False, is_surface=True).convert_list_object())
    s = re.sub("\s+", " ", s).strip()
    return s


def read_langs(en_file, ja_file):
    """ Reads corpuses and returns a Lang object for each language and all normalized sentence pairs.
    """
    segmenter = JumanppWrapper()
    en_lines = open(en_file).read().strip().split("\n")
    ja_lines = open(ja_file).read().strip().split("\n")
    pairs = [[normalize_en(l1), normalize_ja(l2, segmenter)] for l1, l2 in zip(en_lines, ja_lines)]

    en = Lang("en")
    ja = Lang("ja")

    return en, ja, pairs


def filter_pair(p):
    """ Filter out sentences that exceed maximum length.
    """
    return len(p[0].split(" ")) < c.MAX_LENGTH and \
        len(p[1].split(" ")) < c.MAX_LENGTH


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


if __name__ == "__main__":
    en, ja, pairs = read_langs(c.EN_PATH, c.JA_PATH)
    print("Number of sentences:", len(pairs))
    pairs = filter_pairs(pairs)
    print("Number of trimmed sentences:", len(pairs))
    for pair in pairs:
        en.add_sentence(pair[0])
        ja.add_sentence(pair[1])
    print("Number of {} words: {}".format(en.name, en.n_words))
    print("Number of {} words: {}".format(ja.name, ja.n_words))

    pickle.dump(en, open(c.EN_LANG_PATH, "wb"))
    pickle.dump(ja, open(c.JA_LANG_PATH, "wb"))
    pickle.dump(pairs, open(c.PAIRS_PATH, "wb"))
