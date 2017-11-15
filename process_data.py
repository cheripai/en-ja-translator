import constants as c
import multiprocessing as mp
import numpy as np
import pickle
import re
import torch
from JapaneseTokenizer import JumanppWrapper
from Lang import Lang


def normalize_en(s):
    """ Processes an English string by removing non-alphabetical characters (besides .!?).
    """
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^\w.!?]+", r" ", s, flags=re.UNICODE)
    return s


def normalize_ja(s, segmenter):
    """ Processes a Japanese string by removing non-word characters and separating tokens with spaces.
    """
    s = s.strip()
    s = re.sub(r"[^\w.!?。]+", r" ", s, flags=re.UNICODE)
    s = " ".join(segmenter.tokenize(sentence=s, is_feature=False, is_surface=True).convert_list_object())
    s = re.sub("\s+", " ", s).strip()
    return s


def normalize(en_lines, ja_lines):
    """ Process lists of both English and Japanese strings.
    """
    segmenter = JumanppWrapper()
    return [[normalize_en(l1), normalize_ja(l2, segmenter)] for l1, l2 in zip(en_lines, ja_lines)]


def read_langs(en_file, ja_file, n_processes=4):
    """ Reads corpuses and returns a Lang object for each language and all normalized sentence pairs.
    """
    en_lines = open(en_file).read().split("\n")
    ja_lines = open(ja_file).read().split("\n")

    pool = mp.Pool(processes=n_processes)
    interval = len(en_lines) // n_processes
    results = [
        pool.apply_async(
            normalize, args=(en_lines[i * interval:(i + 1) * interval], ja_lines[i * interval:(i + 1) * interval]))
        for i in range(n_processes)
    ]
    pairs = []
    for p in results:
        pairs += p.get()

    en = Lang("en")
    ja = Lang("ja")

    return en, ja, pairs


def filter_pair_by_vocab(p, lang1, lang2):
    """ Filter out sentences if they do not contain words in vocab.
    """
    s1 = p[0].split(" ")
    s2 = p[1].split(" ")
    for word in s1:
        if word not in lang1.word2index:
            return False
    for word in s2:
        if word not in lang2.word2index:
            return False
    return True


def filter_pair_by_len(p, maxlen=c.MAX_LENGTH):
    """ Filter out sentences if they are greater than maximum length.
    """
    return len(p[0].split(" ")) < maxlen and len(p[1].split(" ")) < maxlen


def filter_vocab(lang, min_words=2):
    """ Filters out words from Lang with counts less than min_words in place.
    """
    remove_words = [key for key in lang.word2count.keys() if lang.word2count[key] < min_words]
    for word in remove_words:
        lang.remove_word(word)


def load_en_w2v(fname):
    w2v_en = {}
    with open(fname) as f:
        for line in f:
            line = line.split()
            w = line[0]
            v = np.array([float(x) for x in line[1:]])
            w2v_en[w] = v
    return w2v_en


def load_ja_w2v(fname):
    w2v_ja = {}
    with open(fname) as f:
        for line in f:
            if "[" in line:
                line = line.strip().replace("[", "").split()
                _, w, v = line[0], line[1], line[2:]
            elif "]" in line:
                line = line.strip().replace("]", "").split()
                v += line
                v = np.array([float(val) for val in v])
                w2v_ja[w] = v
            else:
                line = line.strip().split()
                v += line
    return w2v_ja


def build_vecs(vocab, w2v, vec_dim=300):
    vecs = np.zeros((len(vocab), vec_dim))
    not_found = 0
    for i, w in enumerate(vocab):
        try:
            vecs[i] = w2v[w]
        except KeyError:
            not_found += 1
            vecs[i] = np.random.normal(scale=0.6, size=vec_dim)
    print("Did not find {} words".format(not_found))
    return torch.FloatTensor(vecs)


if __name__ == "__main__":
    en, ja, pairs = read_langs(c.EN_PATH, c.JA_PATH)
    print("Number of sentences:", len(pairs))
    pairs = [pair for pair in pairs if filter_pair_by_len(pair, c.MAX_LENGTH)]
    for pair in pairs:
        en.add_sentence(pair[0])
        ja.add_sentence(pair[1])
    filter_vocab(en, c.MIN_VOCAB_WORDS)
    filter_vocab(ja, c.MIN_VOCAB_WORDS)
    pairs = [pair for pair in pairs if filter_pair_by_vocab(pair, en, ja)]
    print("Number of trimmed sentences:", len(pairs))
    print("Number of {} words: {}".format(en.name, en.n_words))
    print("Number of {} words: {}".format(ja.name, ja.n_words))

    w2v_en = load_en_w2v(c.EN_W2V_PATH)
    w2v_ja = load_ja_w2v(c.JA_W2V_PATH)
    en_vecs = build_vecs(list(en.word2index.keys()), w2v_en, vec_dim=300)
    ja_vecs = build_vecs(list(ja.word2index.keys()), w2v_ja, vec_dim=300)

    pickle.dump(en, open(c.EN_LANG_PATH, "wb"))
    pickle.dump(ja, open(c.JA_LANG_PATH, "wb"))
    pickle.dump(pairs, open(c.PAIRS_PATH, "wb"))
    pickle.dump(en_vecs, open(c.EN_VECS_PATH, "wb"))
    pickle.dump(ja_vecs, open(c.JA_VECS_PATH, "wb"))

