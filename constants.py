import torch

BATCH_SIZE = 256
TEST_SIZE = 0.2
TEACHER_FORCING_RATIO = 0.2
MAX_LENGTH = 30
LR = 0.001
HIDDEN_SIZE = 300
DIRECTIONS = 2
LAYERS = 1
DROPOUT = 0.15

SOS_token = 0
EOS_token = 1
MIN_VOCAB_WORDS = 2

EN_PATH = "data/en.txt"
JA_PATH = "data/ja.txt"
EN_W2V_PATH = "data/glove.6B.300d.txt"
JA_W2V_PATH = "data/ja.tsv"
EN_LANG_PATH = "data/en.pkl"
JA_LANG_PATH = "data/ja.pkl"
EN_VECS_PATH = "data/en_vecs.pkl"
JA_VECS_PATH = "data/ja_vecs.pkl"
PAIRS_PATH = "data/en_ja_pairs.pkl"
ENCODER_PATH = "data/encoder.model"
DECODER_PATH = "data/decoder.model"

use_cuda = torch.cuda.is_available()
