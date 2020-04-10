VOCAB_SIZE=16000
#onmt-build-vocab --size $VOCAB_SIZE  --save_vocab data/splitted_data/train/vocab10000.en data/splitted_data/train/train_token10000.en
#onmt-build-vocab --size $VOCAB_SIZE  --save_vocab data/splitted_data/train/vocab10000.fr data/splitted_data/train/train_token10000.fr
python custom_transformer_training.py --src data/splitted_data/train/train_token10000.en --tgt data/splitted_data/train/train_token10000.fr --src_vocab data/splitted_data/train/vocab10000.en --tgt_vocab data/splitted_data/train/vocab10000.fr train
