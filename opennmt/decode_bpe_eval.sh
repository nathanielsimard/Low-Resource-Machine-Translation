mkdir -p run/eval/predictions_bpe
for f in run/eval/predictions.*; do
    # do some stuff here with "$f"
    # remember to quote it or spaces may misbehave
    b=$(basename $f)
    spm_decode --model=../tmp/fr_bpe.model --input_format=piece < $f  > run/eval/predictions_bpe/$b
    BLEU=$(sacrebleu --input=run/eval/predictions_bpe/$b ../tmp/valid.fr --tokenize none --score-only)
    echo BLEU for $b: $BLEU    
done

