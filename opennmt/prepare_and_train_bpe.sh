#https://github.com/google/sentencepiece
#Build and Install SentencePiece
# git clone https://github.com/google/sentencepiece.git
# sentencepiece
# mkdir build
# cd build
# cmake ..
# make -j $(nproc)
# sudo make install
# sudo ldconfig -v
VOCAB_SIZE=4000
bash train_test_split.sh
spm_train --input=../tmp/train.en --model_prefix=../tmp/en_bpe --vocab_size=$VOCAB_SIZE --model_type=bpe
spm_train --input=../tmp/train.fr --model_prefix=../tmp/fr_bpe --vocab_size=$VOCAB_SIZE --model_type=bpe
spm_encode --model=../tmp/en_bpe.model --output_format=piece < ../tmp/train.en  > ../tmp/train_bpe.en
spm_encode --model=../tmp/en_bpe.model --output_format=piece < ../tmp/valid.en  > ../tmp/valid_bpe.en
spm_encode --model=../tmp/en_bpe.model --output_format=piece < ../tmp/test.en  > ../tmp/test_bpe.en
spm_encode --model=../tmp/fr_bpe.model --output_format=piece < ../tmp/train.fr  > ../tmp/train_bpe.fr
spm_encode --model=../tmp/fr_bpe.model --output_format=piece < ../tmp/valid.fr  > ../tmp/valid_bpe.fr
spm_encode --model=../tmp/en_bpe.model --output_format=piece < ../tmp/test.fr  > ../tmp/test_bpe.fr

DATA=data_bpe.yml

onmt-build-vocab --size $VOCAB_SIZE --save_vocab ../tmp/src-vocab.txt ../tmp/train_bpe.en
onmt-build-vocab --size $VOCAB_SIZE --save_vocab ../tmp/tgt-vocab.txt ../tmp/train_bpe.fr

#Models = [--model_type {GPT2Small,ListenAttendSpell,LstmCnnCrfTagger,LuongAttention,NMTBigV1,
#            NMTMediumV1,NMTSmallV1,Transformer,TransformerBase,TransformerBaseRelative,
#            TransformerBig,TransformerBigRelative,TransformerRelative}]

MODEL=Transformer
#mkdir -p run
onmt-main --model_type $MODEL --config $DATA --auto_config train --with_eval
