mkdir -p ../tmp
head -n 10000 ../data/train.lang1 > ../tmp/train.en
head -n 10000 ../data/train.lang2 > ../tmp/train.fr
tail -n 1000 ../data/train.lang1 > ../tmp/test_and_valid.en
tail -n 1000 ../data/train.lang2 > ../tmp/test_and_valid.fr
head -n 500 ../tmp/test_and_valid.en > ../tmp/valid.en
tail -n 500 ../tmp/test_and_valid.en > ../tmp/test.en
head -n 500 ../tmp/test_and_valid.fr > ../tmp/valid.fr
tail -n 500 ../tmp/test_and_valid.fr > ../tmp/test.fr

