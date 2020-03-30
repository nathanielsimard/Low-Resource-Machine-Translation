#https://opennmt.net/OpenNMT-tf/quickstart.html
#mkdir -p data
#cd data
VOCAB_SIZE=20000
EPOCHS=10
DATA=data.yml
#DATA=toy-ende.yml
onmt-build-vocab --size $VOCAB_SIZE --save_vocab ../data/splitted_data/train/src-vocab.txt ../data/splitted_data/train/train_token.en
onmt-build-vocab --size $VOCAB_SIZE --save_vocab ../data/splitted_data/train/tgt-vocab.txt ../data/splitted_data/train/train_token.fr
#Models = [--model_type {GPT2Small,ListenAttendSpell,LstmCnnCrfTagger,LuongAttention,NMTBigV1,
#            NMTMediumV1,NMTSmallV1,Transformer,TransformerBase,TransformerBaseRelative,
#            TransformerBig,TransformerBigRelative,TransformerRelative}]

MODEL=Transformer
onmt-main --model_type $MODEL --config $DATA --auto_config train --with_eval | tee run/opennmt-output.log 

#for i in $(seq 1 $EPOCHS); do
#    echo $i; 
#    onmt-main --model_type $MODEL --config data.yml --auto_config train --with_eval
#done
#

