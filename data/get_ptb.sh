#!/bin/bash

echo "[1/3] Downloading data..."
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
echo "[2/3] Extracting files..."
tar --warning=no-unknown-keyword -xf simple-examples.tgz
rm simple-examples.tgz
mkdir ptb
mv simple-examples/data/ptb.test.txt ptb/test.txt
mv simple-examples/data/ptb.train.txt ptb/train.txt
mv simple-examples/data/ptb.valid.txt ptb/valid.txt
rm -r simple-examples
echo "[3/3] Preprocessing text files..."
mkdir ptb/preprocess
python ../preprocess/preprocess_text.py ptb/ stopwords.txt --bow_vocab_size 2000
#cd ../
#python -c "import data_utils; data_utils.serialize_corpus('data/ptb/preprocess')"
