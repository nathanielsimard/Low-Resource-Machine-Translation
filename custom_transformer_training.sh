VOCAB_SIZE=4000
onmt-build-vocab --size $VOCAB_SIZE  --save_vocab data/splitted_data/train/vocab10000.en data/splitted_data/train/train_token10000.en
onmt-build-vocab --size $VOCAB_SIZE  --save_vocab data/splitted_data/train/vocab10000.fr data/splitted_data/train/train_token10000.fr
