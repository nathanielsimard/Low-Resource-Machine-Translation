from src.model import gru_attention, lstm, lstm_luong_attention, transformer, masked_lm

from src import logging

logger = logging.create_logger(__name__)


def create_lstm(args, input_vocab_size, target_vocab_size):
    return lstm.Lstm(input_vocab_size + 1, target_vocab_size + 1)


def create_transformer(args, input_vocab_size, target_vocab_size):
    model = transformer.Transformer(
        num_layers=2,
        num_heads=2,
        dff=256,
        d_model=256,
        input_vocab_size=input_vocab_size + 1,
        target_vocab_size=target_vocab_size + 1,
        pe_input=input_vocab_size + 1,
        pe_target=target_vocab_size + 1,
        rate=0.1,
    )
    return model


def create_gru_attention(args, input_vocab_size, target_vocab_size):
    return gru_attention.GRU(input_vocab_size + 1, target_vocab_size + 1)


def create_lstm_luong_attention(args, input_vocab_size, target_vocab_size):
    return lstm_luong_attention.LSTM_ATTENTION(
        input_vocab_size + 1, target_vocab_size + 1
    )


def create_demi_bert(args, input_vocab_size, target_vocab_size):
    return masked_lm.DemiBERT(
        num_layers=6,
        embedding_size=256,
        num_heads=8,
        dff=512,
        vocab_size=input_vocab_size,
        max_pe=input_vocab_size,
        dropout=0.1,
    )


MODELS = {
    lstm.NAME: create_lstm,
    transformer.NAME: create_transformer,
    gru_attention.NAME: create_gru_attention,
    lstm_luong_attention.NAME: create_lstm_luong_attention,
    masked_lm.NAME: create_demi_bert,
}


def find(args, input_vocab_size, target_vocab_size):
    try:
        return MODELS[args.model](args, input_vocab_size, target_vocab_size)
    except KeyError as e:
        logger.error(
            f"Model {args.model} is not supported, available models are {list(MODELS.keys())}."
        )
        raise ValueError(e)
