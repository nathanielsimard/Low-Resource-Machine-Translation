import hashlib
import json
from typing import Any, Dict
import copy

from src import logging
from src.model import gru_attention, lstm_luong_attention, masked_lm, transformer

logger = logging.create_logger(__name__)


def read_json_file(file_name: str) -> Dict[str, Any]:
    with open(file_name) as file:
        return json.load(file)


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
        "vocab_size": input_vocab_size + 1,
        "max_pe": input_vocab_size + 1,
        **read_json_file(args.hyperparameters),
    }
    return masked_lm.DemiBERT(**hyperparameters), hyperparameters


def create_transformer_pretrained(args, input_vocab_size, target_vocab_size):
    demi_bert_args = copy.deepcopy(args)
    demi_bert_args.model = "demi-bert"
    demi_bert = find(demi_bert_args, input_vocab_size, target_vocab_size)
    demi_bert.load("2")
    hyperparameters = {
        "input_vocab_size": input_vocab_size + 1,
        "target_vocab_size": target_vocab_size,
        "pe_input": input_vocab_size + 1,
        "pe_target": target_vocab_size,
        **read_json_file(args.hyperparameters),
    }
    args.model = "transformer"
    transformer_ = transformer.Transformer(**hyperparameters)
    transformer_.encoder = demi_bert.encoder
    transformer_.title += "-pretrained"
    return transformer_, hyperparameters


MODELS = {
    transformer.NAME: create_transformer,
    gru_attention.NAME: create_gru_attention,
    lstm_luong_attention.NAME: create_lstm_luong_attention,
    masked_lm.NAME: create_demi_bert,
    "transformer-pretrained": create_transformer_pretrained,
}


def _hyperparameters_string(hyperparameters):
    # Make sure every key is in order to load the
    # good model with the md5.
    keys = list(hyperparameters.keys())
    keys.sort()

    string = "{"
    for key in keys:
        string += f"\n\t{key}: {hyperparameters[key]}"
    string += "\n}"

    return string


def find(args, input_vocab_size, target_vocab_size):
    try:
        model, hyperparameters = MODELS[args.model](
            args, input_vocab_size, target_vocab_size
        )

        # Usefull to not override the same model with different hyperparameters.
        hyperparameters = _hyperparameters_string(hyperparameters)
        model_id = hashlib.md5(str.encode(hyperparameters)).hexdigest()
        model.title += "-" + str(model_id)

        logger.info(f"Model {model.title} with hyperparameters: {hyperparameters}")
        return model
    except KeyError as e:
        logger.error(
            f"Model {args.model} is not supported, available models are {list(MODELS.keys())}."
        )
        raise ValueError(e)
