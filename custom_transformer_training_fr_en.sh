#In order to prepare a model for back-translation. 
#Using only BPE. Monolingual data was not tokenized properly and
#should not be used until this is fixed.
python custom_transformer_training.py \
 --src data/splitted_data/train/train_token10000.fr \
 --valsrc data/splitted_data/valid/val_token10000.fr \
 --tgt data/splitted_data/train/train_token10000.en \
 --valtgt data/splitted_data/valid/val_token10000.en \
 --bpe train

shuf data/token_unaligned/unaligned.fr > unaligned_fr_suffled.tmp
head unaligned_fr_suffled.tmp -n 20000 > unaligned_fr.tmp

python custom_transformer_training.py \
 --src unaligned_fr.tmp \
 --bpe translate
 
 #Save the back-translated english sentences
mv output.txt.decoded bt.en.tmp

#Train with BT and BPE
python custom_transformer_training.py \
 --src data/splitted_data/train/train_token10000.en \
 --valsrc data/splitted_data/valid/val_token10000.en \
 --tgt data/splitted_data/train/train_token10000.fr \
 --valtgt data/splitted_data/valid/val_token10000.fr \
 --btsrc bt.en.tmp \
 --bttgt unaligned_fr.tmp \
 --bpe train
 