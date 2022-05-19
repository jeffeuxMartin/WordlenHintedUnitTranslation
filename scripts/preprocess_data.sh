#!sh
##########################3 \u{1F644}##########################3 \u{1F644}##########################3 \u{1F644}##########################3 \u{1F644}##########################3 \u{1F644}##########################3 \u{1F644}##########################3 \u{1F644}##########################3 \u{1F644}
#!zsh
# try_fairseq_dummy
# "mylogs_dummy`date +%H%M%S`"
# "mysaveddir_dummy`date +%H%M%S`"

export PATH=`
    `/home/jeffeuxmartin/miniconda3/bin:`
    `/home/jeffeuxmartin/.local/bin:`
    `:"$PATH"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/jeffeuxmartin/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/jeffeuxmartin/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/jeffeuxmartin/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/jeffeuxmartin/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate fairseq_env
##########################3 ##########################3 ##########################3 ##########################3 ##########################3 ##########################3 ##########################3 ##########################3 

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
SRCDIR="${SRCDIR:=collunits}" ; SRC="${SRC:=collunit}"
TGTDIR="${TGTDIR:=texts}"     ; TGT="${TGT:=en}"
LENDIR="${LENDIR:=lengths}"   ; LEN="${LEN:=len}"

mkdir -p $Unbinarized_dataset_path
mkdir -p $Binarized_dataset_path

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
   $Binarized_dataset_path/train.$SRC-$TGT.len
   
cp $Unbinarized_dataset_path/dev.$LEN \
   $Binarized_dataset_path/valid.$SRC-$TGT.len  # NOTE!
   
cp $Unbinarized_dataset_path/test.$LEN \
   $Binarized_dataset_path/test.$SRC-$TGT.len
# FIXME: bad! 不要硬規定死 .len 比較好！
