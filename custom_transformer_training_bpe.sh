#VOCAB_SIZE=20000
#onmt-build-vocab --size $VOCAB_SIZE  --save_vocab data/splitted_data/train/vocab10000.en data/splitted_data/train/train_token10000.en
#onmt-build-vocab --size $VOCAB_SIZE  --save_vocab data/splitted_data/train/vocab10000.fr data/splitted_data/train/train_token10000.fr
python opennmt_transformer.py \
 --src data/splitted_data/train/train_token10000.en \
 --valsrc data/splitted_data/valid/val_token10000.en \
 --tgt data/splitted_data/train/train_token10000.fr \
 --valtgt data/splitted_data/valid/val_token10000.fr \
 --bpe train
#--validate_now \

 