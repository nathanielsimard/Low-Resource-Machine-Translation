import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict

from src import logging
from src.model import (gru_attention, lstm, lstm_luong_attention, masked_lm,
                       transformer)

logger = logging.create_logger(__name__)


def read_json_file(file_name: str) -> Dict[str, Any]:
    with open(file_name) as file:
        return json.load(file)


def create_lstm(args, input_vocab_size, target_vocab_size):
    hyperparameters = {
        "input_vocab_size": input_vocab_size + 1,
        "output_vocab_size": target_vocab_size + 1,
        **read_json_file(args.hyperparameters),
    }
    return lstm.Lstm(**hyperparameters), hyperparameters


def create_transformer(args, input_vocab_size, target_vocab_size):
    hyperparameters = {
        "input_vocab_size": input_vocab_size + 1,
        "target_vocab_size": target_vocab_size + 1,
        "pe_input": input_vocab_size + 1,
        "pe_target": target_vocab_size + 1,
        **read_json_file(args.hyperparameters),
    }
    return transformer.Transformer(**hyperparameters), hyperparameters


def create_gru_attention(args, input_vocab_size, target_vocab_size):
    hyperparameters = {
        "input_vocab_size": input_vocab_size + 1,
        "output_vocab_size": target_vocab_size + 1,
        **read_json_file(args.hyperparameters),
    }
    return gru_attention.GRU(**hyperparameters), hyperparameters


def create_lstm_luong_attention(args, input_vocab_size, target_vocab_size):
    hyperparameters = {
        "input_vocab_size": input_vocab_size + 1,
        "output_vocab_size": target_vocab_size + 1,
    }
    return lstm_luong_attention.LSTM_ATTENTION(**hyperparameters), hyperparameters


def create_demi_bert(args, input_vocab_size, target_vocab_size):
    hyperparameters = {
        "vocab_size": input_vocab_size,
        "max_pe": input_vocab_size,
        **read_json_file(args.hyperparameters),
    }
    return masked_lm.DemiBERT(**hyperparameters), hyperparameters


MODELS = {
    lstm.NAME: create_lstm,
    transformer.NAME: create_transformer,
    gru_attention.NAME: create_gru_attention,
    lstm_luong_attention.NAME: create_lstm_luong_attention,
    masked_lm.NAME: create_demi_bert,
}


def _print_hyperparameters(hyperparameters):
    string = "{"
    for key, value in hyperparameters.items():
        string += f"\n\t{key}: {value}"
    string += "\n}"

    return string


def find(args, input_vocab_size, target_vocab_size):
    try:
        model, hyperparameters = MODELS[args.model](
            args, input_vocab_size, target_vocab_size
        )

        # Usefull to not override the same model with different hyperparameters.
        hyperparameters = OrderedDict(hyperparameters)
        model_id = hashlib.md5(str.encode(str(hyperparameters))).hexdigest()
        model.title += "-" + str(model_id)

        logger.info(
            f"Model {model.title} with hyperparameters: {_print_hyperparameters(hyperparameters)}"
        )
        return model
    except KeyError as e:
        logger.error(
            f"Model {args.model} is not supported, available models are {list(MODELS.keys())}."
        )
        raise ValueError(e)
