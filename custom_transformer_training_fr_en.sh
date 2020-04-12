#In order to prepare a model for back-translation. 
#Using only BPE. Monolingual data was not tokenized properly and
#should not be used until this is fixed.
python custom_transformer_training.py \
 --src data/splitted_data/train/train_token10000.fr \
 --valsrc data/splitted_data/valid/val_token10000.fr \
 --tgt data/splitted_data/train/train_token10000.en \
 --valtgt data/splitted_data/valid/val_token10000.en \
 --bpe train


 