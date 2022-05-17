#!sh

Original_dataset_path=`   `data/LibriSpeechUnits
Unbinarized_dataset_path=``data/FairseqLibriUnits
Binarized_dataset_path=`  `data/BinFairseqLibriUnits

TRAIN_SPLIT=``train-clean-100
DEV_SPLIT=`  `dev-clean
TEST_SPLIT=` `test-clean

SRCDIR=collunits ; SRC=collunit
TGTDIR=texts     ; TGT=en
LENDIR=lengths   ; LEN=len

cp $Original_dataset_path/$SRCDIR/$TRAIN_SPLIT.$SRC \
   $Unbinarized_dataset_path/train.$SRC
cp $Original_dataset_path/$TGTDIR/$TRAIN_SPLIT.$TGT \
   $Unbinarized_dataset_path/train.$TGT

cp $Original_dataset_path/$SRCDIR/$DEV_SPLIT.$SRC \
   $Unbinarized_dataset_path/dev.$SRC
cp $Original_dataset_path/$TGTDIR/$DEV_SPLIT.$TGT \
   $Unbinarized_dataset_path/dev.$TGT

cp $Original_dataset_path/$SRCDIR/$TEST_SPLIT.$SRC \
   $Unbinarized_dataset_path/test.$SRC
cp $Original_dataset_path/$TGTDIR/$TEST_SPLIT.$TGT \
   $Unbinarized_dataset_path/test.$TGT
   
cp $Original_dataset_path/$LENDIR/$TRAIN_SPLIT.$LEN \
   $Unbinarized_dataset_path/train.$LEN
cp $Original_dataset_path/$LENDIR/$DEV_SPLIT.$LEN \
   $Unbinarized_dataset_path/dev.$LEN
cp $Original_dataset_path/$LENDIR/$TEST_SPLIT.$LEN \
   $Unbinarized_dataset_path/test.$LEN
  
fairseq-preprocess \
-s $SRC \
-t $TGT \
--trainpref `  `$Unbinarized_dataset_path/train` ` \
--validpref `  `$Unbinarized_dataset_path/dev`   ` \
--testpref `   `$Unbinarized_dataset_path/test`  ` \
--destdir $Binarized_dataset_path \
--workers 8

cp $Unbinarized_dataset_path/train.$LEN \
   $Binarized_dataset_path/train.$SRC-$TGT.$LEN
   
cp $Unbinarized_dataset_path/dev.$LEN \
   $Binarized_dataset_path/dev.$SRC-$TGT.$LEN
   
cp $Unbinarized_dataset_path/test.$LEN \
   $Binarized_dataset_path/test.$SRC-$TGT.$LEN
