import torch

BATCH_SIZE = 256
TEST_SIZE = 0.2
TEACHER_FORCING_RATIO = 0.3
MAX_LENGTH = 30
LR = 0.001
HIDDEN_SIZE = 512
DIRECTIONS = 2
LAYERS = 1
DROPOUT = 0.1

SOS_token = 0
EOS_token = 1

EN_PATH = "data/en.txt"
JA_PATH = "data/ja.txt"
EN_LANG_PATH = "data/en.pkl"
JA_LANG_PATH = "data/ja.pkl"
PAIRS_PATH = "data/en_ja_pairs.pkl"
ENCODER_PATH = "data/encoder.model"
DECODER_PATH = "data/decoder.model"

use_cuda = torch.cuda.is_available()
