#!sh

# === args === #
Original_dataset_path=`   `"${Original_dataset_path:=data/LibriSpeechUnits}"
Unbinarized_dataset_path=``"${Unbinarized_dataset_path:=data/FairseqLibriUnits}"
Binarized_dataset_path=`  `"${Binarized_dataset_path:=data/BinFairseqLibriUnits}"
UNITDICTPATH=`            `"${UNITDICTPATH:=data/dummy_collunit.dict}"

TRAIN_SPLIT=`             `"${TRAIN_SPLIT:=train-clean-100}"
DEV_SPLIT=`               `"${DEV_SPLIT:=dev-clean}"
TEST_SPLIT=`              `"${TEST_SPLIT:=test-clean}"

# ref.: https://stackoverflow.com/questions/2953646/how-can-i-declare-and-use-boolean-variables-in-a-shell-script
LOWERCASE="${LOWERCASE:=false}"

# --- --- #
mkdir -p $Unbinarized_dataset_path
mkdir -p $Binarized_dataset_path

SRCDIR=collunits ; SRC=collunit
TGTDIR=texts     ; TGT=en
LENDIR=lengths   ; LEN=len

cp $Original_dataset_path/$SRCDIR/$TRAIN_SPLIT.$SRC \
   $Unbinarized_dataset_path/train.$SRC
if [ "$LOWERCASE" = true ]; then
   # ref.: https://stackoverflow.com/questions/2264428/how-to-convert-a-string-to-lower-case-in-bash
   cat $Original_dataset_path/$TGTDIR/$TRAIN_SPLIT.$TGT \
    | tr '[:upper:]' '[:lower:]' \
    > $Unbinarized_dataset_path/train.$TGT
else
   cp $Original_dataset_path/$TGTDIR/$TRAIN_SPLIT.$TGT \
      $Unbinarized_dataset_path/train.$TGT
fi

cp $Original_dataset_path/$SRCDIR/$DEV_SPLIT.$SRC \
   $Unbinarized_dataset_path/dev.$SRC
if [ "$LOWERCASE" = true ]; then
   cat $Original_dataset_path/$TGTDIR/$DEV_SPLIT.$TGT \
    | tr '[:upper:]' '[:lower:]' \
    > $Unbinarized_dataset_path/dev.$TGT
else
   cp $Original_dataset_path/$TGTDIR/$DEV_SPLIT.$TGT \
      $Unbinarized_dataset_path/dev.$TGT
fi

cp $Original_dataset_path/$SRCDIR/$TEST_SPLIT.$SRC \
   $Unbinarized_dataset_path/test.$SRC
if [ "$LOWERCASE" = true ]; then
   cat $Original_dataset_path/$TGTDIR/$TEST_SPLIT.$TGT \
    | tr '[:upper:]' '[:lower:]' \
    > $Unbinarized_dataset_path/test.$TGT
else
   cp $Original_dataset_path/$TGTDIR/$TEST_SPLIT.$TGT \
      $Unbinarized_dataset_path/test.$TGT
fi
   
cp $Original_dataset_path/$LENDIR/$TRAIN_SPLIT.$LEN \
   $Unbinarized_dataset_path/train.$LEN
cp $Original_dataset_path/$LENDIR/$DEV_SPLIT.$LEN \
   $Unbinarized_dataset_path/dev.$LEN
cp $Original_dataset_path/$LENDIR/$TEST_SPLIT.$LEN \
   $Unbinarized_dataset_path/test.$LEN
  
fairseq-preprocess \
-s $SRC --srcdict $UNITDICTPATH \
-t $TGT \
--trainpref `  `$Unbinarized_dataset_path/train` ` \
--validpref `  `$Unbinarized_dataset_path/dev`   ` \
--testpref `   `$Unbinarized_dataset_path/test`  ` \
--destdir $Binarized_dataset_path \
--workers $(python -c 'import os; print(len(os.sched_getaffinity(0)))')

cp $Unbinarized_dataset_path/train.$LEN \
   $Binarized_dataset_path/train.$SRC-$TGT.$LEN
   
cp $Unbinarized_dataset_path/dev.$LEN \
   $Binarized_dataset_path/valid.$SRC-$TGT.$LEN  # NOTE!
   
cp $Unbinarized_dataset_path/test.$LEN \
   $Binarized_dataset_path/test.$SRC-$TGT.$LEN
