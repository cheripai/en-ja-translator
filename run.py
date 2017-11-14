import constants as c
import pickle
import torch
from Lang import Lang, sentence2indexes
from models.EncoderDecoder import Encoder, Decoder
from process_data import normalize_en
from torch.autograd import Variable


def evaluate(encoder, decoder, sentence, max_length=30):
    input_variable = Variable(sentence2indexes(input_lang, sentence))
    input_length = input_variable.size()[0]

    encoder_hidden = encoder.init_hidden(1)
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size * encoder.directions))
    encoder_outputs = encoder_outputs.cuda() if c.use_cuda else encoder_outputs

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[i], encoder_hidden)
        encoder_outputs[i] = encoder_outputs[i] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[c.SOS_token]]))
    decoder_input = decoder_input.cuda() if c.use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    encoder_outputs = encoder_outputs.unsqueeze(0)

    decoded_words = []

    for i in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == c.EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if c.use_cuda else decoder_input

    return decoded_words


if __name__ == "__main__":
    try:
        input_lang = pickle.load(open(c.EN_LANG_PATH, "rb"))
        output_lang = pickle.load(open(c.JA_LANG_PATH, "rb"))
    except FileNotFoundError:
        raise Exception("Could not find {} or {}. Have you run process_data.py?".format(c.EN_LANG_PATH, c.JA_LANG_PATH))

    encoder = Encoder(
        input_lang.n_words,
        c.HIDDEN_SIZE,
        emb_weights=None,
        directions=c.DIRECTIONS,
        n_layers=c.LAYERS,
        dropout_p=c.DROPOUT)
    decoder = Decoder(
        c.HIDDEN_SIZE,
        output_lang.n_words,
        emb_weights=None,
        directions=c.DIRECTIONS,
        n_layers=c.LAYERS,
        dropout_p=c.DROPOUT)

    if c.use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    try:
        encoder.load_state_dict(torch.load(c.ENCODER_PATH))
        decoder.load_state_dict(torch.load(c.DECODER_PATH))
    except FileNotFoundError:
        raise Exception("Could not find model weights. Have you run train.py?")

    while True:
        input_string = normalize_en(input("> "))
        s = evaluate(encoder, decoder, input_string)[:-1]
        print("".join(s))
