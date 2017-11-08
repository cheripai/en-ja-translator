import constants as c
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset


class Lang:
    """" Holds vocabulary for each language and dictionaries to convert words to and from indexes
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.word2count = {"SOS": 0, "EOS": 0}
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

    def remove_word(self, word):
        if word != "SOS" and word != "EOS":
            self.n_words -= 1
            idx = self.word2index[word]
            last_word = self.index2word[self.n_words]

            del self.word2index[word]
            del self.word2count[word]
            del self.index2word[self.n_words]

            if last_word != word:
                self.word2index[last_word] = idx
                self.index2word[idx] = last_word


class LangDataset(Dataset):
    """ Subclass of Pytorch Dataset.
        Holds sentence data from both languages.
    """
    def __init__(self, pairs, input_lang, output_lang, max_length=30):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]

    def collate(self, batch):
        batch_size = len(batch)
        input_train = torch.LongTensor(batch_size, self.max_length).zero_()
        target_train = torch.LongTensor(batch_size, self.max_length).zero_()

        if c.use_cuda:
            input_train = input_train.cuda()
            target_train = target_train.cuda()

        for i, pair in enumerate(batch):
            input_train[i] = sentence2indexes(self.input_lang, pair[0], self.max_length)
            target_train[i] = sentence2indexes(self.output_lang, pair[1], self.max_length)

        return Variable(input_train), Variable(target_train)


def sentence2indexes(lang, sentence, max_length=30):
    indexes = [lang.word2index[word] for word in sentence.split(" ")]
    result = torch.LongTensor(max_length)
    result[:] = c.EOS_token
    for i, index in enumerate(indexes):
        result[i] = index

    if c.use_cuda:
        return result.cuda()
    else:
        return result
