cd ../
git clone https://github.com/StonyBrookNLP/ircot.git
cd ircot

bash ./download/raw_data.sh

mv raw_data ../data/corpus
cd ../data/corpus

wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gunzip psgs_w100.tsv.gz