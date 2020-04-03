#Script to run the basic OpenNMT Transformer Custom Model
#(Just to make sure it give the same performance as the command line)

#Train/Valid/Test split
python -c 'from src import utils; utils.split_joint_data("data/train.lang1", "data/train.lang2", "/tmp")'

#Build joint vocabulary:
VOCAB_SIZE=4000
cat /tmp/train.en /tmp/train.fr > /tmp/train.enfr
spm_train --input=/tmp/train.enfr --model_prefix=/tmp/enfr_bpe --vocab_size=$VOCAB_SIZE --model_type=bpe

#Perform Byte-Pair Encoding
spm_encode --model=/tmp/enfr_bpe.model --output_format=piece < /tmp/train.en  > /tmp/train_bpe.en
spm_encode --model=/tmp/enfr_bpe.model --output_format=piece < /tmp/valid.en  > /tmp/valid_bpe.en
spm_encode --model=/tmp/enfr_bpe.model --output_format=piece < /tmp/test.en   > /tmp/test_bpe.en
spm_encode --model=/tmp/enfr_bpe.model --output_format=piece < /tmp/train.fr  > /tmp/train_bpe.fr
spm_encode --model=/tmp/enfr_bpe.model --output_format=piece < /tmp/valid.fr  > /tmp/valid_bpe.fr
spm_encode --model=/tmp/enfr_bpe.model --output_format=piece < /tmp/test.fr   > /tmp/test_bpe.fr

cat /tmp/train_bpe.en /tmp/train_bpe.fr > /tmp/train_bpe.enfr

onmt-build-vocab --size $VOCAB_SIZE --save_vocab /tmp/joint-vocab.txt /tmp/train_bpe.enfr

#We're ready to roll
python run_experiment_opennmt.py --src=/tmp/train_bpe.en --tgt=/tmp/train_bpe.fr --src_vocab=/tmp/joint-vocab.txt --tgt_vocab=/tmp/joint-vocab.txt train

