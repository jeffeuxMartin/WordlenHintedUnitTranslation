#!zsh
ln -s /home/jeffeuxmartin/FairseqAudioWords/data/BinFairseqLibriAE/preprocess.log
ln -s /home/jeffeuxmartin/FairseqAudioWords/data/BinFairseqLibriAE/dict.collunit.txt
ln -s /home/jeffeuxmartin/FairseqAudioWords/data/BinFairseqLibriAE/train.collunit-collunit.collunit.bin
ln -s /home/jeffeuxmartin/FairseqAudioWords/data/BinFairseqLibriAE/train.collunit-collunit.collunit.idx
ln -s /home/jeffeuxmartin/FairseqAudioWords/data/BinFairseqLibriAE/valid.collunit-collunit.collunit.bin
ln -s /home/jeffeuxmartin/FairseqAudioWords/data/BinFairseqLibriAE/valid.collunit-collunit.collunit.idx
ln -s /home/jeffeuxmartin/FairseqAudioWords/data/BinFairseqLibriAE/test.collunit-collunit.collunit.bin
ln -s /home/jeffeuxmartin/FairseqAudioWords/data/BinFairseqLibriAE/test.collunit-collunit.collunit.idx
ln -s /home/jeffeuxmartin/FairseqAudioWords/data/BinFairseqLibriAE/train.collunit-collunit.len
ln -s /home/jeffeuxmartin/FairseqAudioWords/data/BinFairseqLibriAE/valid.collunit-collunit.len
ln -s /home/jeffeuxmartin/FairseqAudioWords/data/BinFairseqLibriAE/test.collunit-collunit.len

for i in /home/jeffeuxmartin/FairseqAudioWords/data/BinFairseqLibriAE/*.len; do cp $i .; done

for LENNAME in *.len; do
TOTAL=$(cat $LENNAME | wc -l); for i in {1..$TOTAL}; do echo 1; done > $LENNAME
done
