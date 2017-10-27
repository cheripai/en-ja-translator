import torch

use_cuda = torch.cuda.is_available()

MAX_LENGTH = 30
SOS_token = 0
EOS_token = 1
