paste ../data/train.lang1 ../data/train.lang2 -d "\\t" | shuf > ../tmp/two_languages_shuffled.txt
head ../tmp/two_languages_shuffled.txt -n 10000 > ../tmp/two_languages_shuffled_train.txt
cat ../tmp/two_languages_shuffled.txt | sed -r "/.{256,}/d" > ../tmp/two_languages_shuffled_train_clipped.txt


