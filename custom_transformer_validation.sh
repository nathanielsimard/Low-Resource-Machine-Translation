VOCAB_SIZE=4000
#onmt-build-vocab --size $VOCAB_SIZE  --save_vocab data/splitted_data/train/vocab10000.en data/splitted_data/train/train_token10000.en
#onmt-build-vocab --size $VOCAB_SIZE  --save_vocab data/splitted_data/train/vocab10000.fr data/splitted_data/train/train_token10000.fr
export TF_ADDONS_PY_OPS=1
python opennmt_transformer.py --src data/splitted_data/valid/val_token10000.fr --src_vocab data/splitted_data/train/vocab10000.en --tgt_vocab data/splitted_data/train/vocab10000.fr translate

