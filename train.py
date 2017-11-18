import constants as c
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from Lang import Lang, LangDataset
from models.EncoderDecoder import Encoder, Decoder
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def train(input_var,
          target_var,
          encoder,
          decoder,
          encoder_opt,
          decoder_opt,
          criterion,
          max_length=30,
          teacher_forcing_ratio=0.5,
          train=True):
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()

    batch_size = input_var.size()[0]
    input_length = input_var.size()[1]
    target_length = target_var.size()[1]

    encoder_hidden = encoder.init_hidden(batch_size)
    encoder_outputs = Variable(torch.zeros(batch_size, max_length, encoder.hidden_size * encoder.directions))
    encoder_outputs = encoder_outputs.cuda() if c.use_cuda else encoder_outputs

    loss = 0

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_var[:, i], encoder_hidden)
        encoder_outputs[:, i] = encoder_output

    decoder_input = Variable(torch.LongTensor(batch_size, 1))
    decoder_input[:] = c.SOS_token
    decoder_input = decoder_input.cuda() if c.use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    if train:
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    else:
        use_teacher_forcing = False

    # Teacher forcing: Feed the target as the next input
    if use_teacher_forcing:
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_var[:, i])
            decoder_input = target_var[:, i]  # Teacher forcing

    # No teacher forcing: use its own predictions as the next input
    else:
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_var[:, i])

            decoder_input = Variable(torch.LongTensor(batch_size, 1))
            decoder_input = decoder_input.cuda() if c.use_cuda else decoder_input
            topv, topi = decoder_output.data.topk(1)
            decoder_input[:] = topi

    if train:
        loss.backward()
        encoder_opt.step()
        decoder_opt.step()

    return loss.data[0] / target_length


def train_iters(encoder, decoder, trainloader, validloader, epochs, lr=0.001):
    encoder_opt = optim.Adam(encoder.parameters(), lr=lr)
    decoder_opt = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    teacher_forcing_ratio = c.TEACHER_FORCING_RATIO
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))

        # Train networks
        teacher_forcing_ratio /= 2
        total_loss = 0
        for i, (input_var, target_var) in enumerate(trainloader):
            loss = train(input_var, target_var, encoder, decoder, encoder_opt, decoder_opt, criterion, teacher_forcing_ratio=teacher_forcing_ratio, train=True)
            total_loss += loss

        avg_loss = total_loss / (i + 1)
        print("Train Loss %.4f" % (avg_loss))

        # Evaluate networks
        total_loss = 0
        for i, (input_var, target_var) in enumerate(validloader):
            loss = train(input_var, target_var, encoder, decoder, encoder_opt, decoder_opt, criterion, teacher_forcing_ratio=0, train=False)
            total_loss += loss

        avg_loss = total_loss / (i + 1)
        print("Valid Loss %.4f" % (avg_loss))


if __name__ == "__main__":
    en = pickle.load(open(c.EN_LANG_PATH, "rb"))
    ja = pickle.load(open(c.JA_LANG_PATH, "rb"))
    en_vecs = pickle.load(open(c.EN_VECS_PATH, "rb"))
    ja_vecs = pickle.load(open(c.JA_VECS_PATH, "rb"))

    pairs = pickle.load(open(c.PAIRS_PATH, "rb"))

    train_pairs, valid_pairs = train_test_split(pairs, test_size=c.TEST_SIZE)
    trainset = LangDataset(train_pairs, en, ja)
    validset = LangDataset(valid_pairs, en, ja)
    trainloader = DataLoader(trainset, batch_size=c.BATCH_SIZE, shuffle=True, collate_fn=trainset.collate)
    validloader = DataLoader(validset, batch_size=c.BATCH_SIZE, shuffle=True, collate_fn=validset.collate)

    encoder = Encoder(
        en.n_words,
        c.HIDDEN_SIZE,
        emb_weights=en_vecs,
        directions=c.DIRECTIONS,
        n_layers=c.LAYERS,
        dropout_p=c.DROPOUT)
    decoder = Decoder(
        c.HIDDEN_SIZE,
        ja.n_words,
        emb_weights=ja_vecs,
        directions=c.DIRECTIONS,
        n_layers=c.LAYERS,
        dropout_p=c.DROPOUT)

    if c.use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    print("Training...")
    train_iters(encoder, decoder, trainloader, validloader, 15, lr=c.LR)

    torch.save(encoder.state_dict(), c.ENCODER_PATH)
    torch.save(decoder.state_dict(), c.DECODER_PATH)
