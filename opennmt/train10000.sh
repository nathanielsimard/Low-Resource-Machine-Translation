#https://opennmt.net/OpenNMT-tf/quickstart.html
#mkdir -p data
#cd data
bash train_test_split.sh
VOCAB_SIZE=20000
EPOCHS=10
onmt-build-vocab --size $VOCAB_SIZE --save_vocab ../tmp/src-vocab.txt ../tmp/train.en
onmt-build-vocab --size $VOCAB_SIZE --save_vocab ../tmp/tgt-vocab.txt ../tmp/train.fr

#Models = [--model_type {GPT2Small,ListenAttendSpell,LstmCnnCrfTagger,LuongAttention,NMTBigV1,
#            NMTMediumV1,NMTSmallV1,Transformer,TransformerBase,TransformerBaseRelative,
#            TransformerBig,TransformerBigRelative,TransformerRelative}]

MODEL=Transformer
onmt-main --model_type $MODEL --config data10000.yml --auto_config train --with_eval

#for i in $(seq 1 $EPOCHS); do
#    echo $i; 
#    onmt-main --model_type $MODEL --config data.yml --auto_config train --with_eval
#done
#

