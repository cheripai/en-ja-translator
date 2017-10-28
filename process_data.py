import constants as c
import re
from JapaneseTokenizer import JumanppWrapper

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word in self.word2count:
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1


def normalize_en(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w.!?]+", r" ", s, flags=re.UNICODE)
    return s


def normalize_ja(s, segmenter):
    s = re.sub(r"[^\w.!?。]+", r" ", s, flags=re.UNICODE)
    s = " ".join(segmenter.tokenize(sentence=s, is_feature=False, is_surface=True).convert_list_object())
    s = re.sub("\s+", " ", s).strip()
    return s


def read_langs(en_file, ja_file):
    segmenter = JumanppWrapper()
    en_lines = open(en_file).read().strip().split("\n")
    ja_lines = open(ja_file).read().strip().split("\n")
    pairs = [[normalize_en(l1), normalize_ja(l2, segmenter)] for l1, l2 in zip(en_lines, ja_lines)]

    en = Lang("en")
    ja = Lang("ja")

    return en, ja, pairs


def filter_pair(p):
    return len(p[0].split(" ")) < c.MAX_LENGTH and \
        len(p[1].split(" ")) < c.MAX_LENGTH


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(en_file, ja_file):
    en, ja, pairs = read_langs(en_file, ja_file)
    print("Number of sentences:", len(pairs))
    pairs = filter_pairs(pairs)
    print("Number of trimmed sentences:", len(pairs))
    for pair in pairs:
        en.add_sentence(pair[0])
        ja.add_sentence(pair[1])
    print(en.name, en.n_words)
    print(ja.name, ja.n_words)
    return en, ja, pairs


if __name__ == "__main__":
    en, ja, pairs = prepare_data("data/en.txt", "data/ja.txt")
    pickle.dump(en, open("data/en.pkl", "wb"))
    pickle.dump(ja, open("data/ja.pkl", "wb"))
    pickle.dump(pairs, open("data/en_ja_pairs.pkl", "wb"))
