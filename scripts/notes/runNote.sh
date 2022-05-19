# silly .txt !!!
TGT=en LENDIR=wordlengths LEN=wordlen \
  DATAPATH=data/BinFairseqLibriUnits \
  MAXTOKENS=4096 LR=2e-4 EPOCHS=200 \
  WANDBPROJ=LS_ASR LOG_FILE=ASR_first SAVE_DIR=ASR_try1st \
      hrun -s -c 16 -m 32 \
      zsh WordlenHintedUnitTranslation/scripts/preprocess_data.sh

# # # # # # # # # # # # # # # # # # # # # # # # # # #
DATAPATH=data/BinFairseqLibriUnits \
  MAXTOKENS=$((40960 / 4)) LR=2e-4 EPOCHS=200 \
  WANDBPROJ=LS_ASR LOG_FILE=ASR_Second__2eNeg4 SAVE_DIR=ASR_try2nd__2eNeg4 \
      hrun -s -c 16 -m 32 -GGGG -g 3080Ti \
      zsh WordlenHintedUnitTranslation/scripts/run_battleship.sh

# DATAPATH=data/BinFairseqLibriUnits \
#   MAXTOKENS=20480 LR=2e-3 EPOCHS=200 \
#   WANDBPROJ=LS_ASR LOG_FILE=ASR_Second__2eNeg3 SAVE_DIR=ASR_try2nd__2eNeg3 \
#       hrun -s -c 16 -m 32 -GG -g 3090 \
#       zsh WordlenHintedUnitTranslation/scripts/run_battleship.sh

# DATAPATH=data/BinFairseqLibriUnits \
#   MAXTOKENS=20480 LR=2e-2 EPOCHS=200 \
#   WANDBPROJ=LS_ASR LOG_FILE=ASR_Second__2eNeg2 SAVE_DIR=ASR_try2nd__2eNeg2 \
#       hrun ` ` -c 16 -m 32 -GG -g TITANRTX \
#       zsh WordlenHintedUnitTranslation/scripts/run_battleship.sh

# DATAPATH=data/BinFairseqLibriUnits \
#   MAXTOKENS=20480 LR=2e-5 EPOCHS=200 \
#   WANDBPROJ=LS_ASR LOG_FILE=ASR_Second__2eNeg5 SAVE_DIR=ASR_try2nd__2eNeg5 \
#       hrun ` ` -c 16 -m 32 -GG -g TITANRTX \
#       zsh WordlenHintedUnitTranslation/scripts/run_battleship.sh

DATAPATH=data/BinFairseqLibriUnits \
  MAXTOKENS=20480 LR=2e-4 EPOCHS=200 \
  WANDBPROJ=LS_ASR LOG_FILE=ASR_Second__2eNeg4 SAVE_DIR=ASR_try2nd__2eNeg4 \
      hrun -s -c 16 -m 32 -GG -g 3090 \
      zsh WordlenHintedUnitTranslation/scripts/run_battleship.sh

# ---------------------

Unbinarized_dataset_path=data/FairseqLibriAE \
Binarized_dataset_path=data/BinFairseqLibriAE \
TGTDIR=collunits TGT=collunit \
LENDIR=wordlengths LEN=wordlen \
hrun -s -c 16 -m 32 \
zsh WordlenHintedUnitTranslation/scripts/preprocess_data.sh

DATAPATH=data/BinFairseqLibriAE \
  MAXTOKENS=$((20480 / 2)) LR=2e-4 EPOCHS=200 \
  WANDBPROJ=LS_AE LOG_FILE=AE__2eNeg4 SAVE_DIR=AE_try1st__2eNeg4 \
      hrun -s -c 16 -m 32 -GGGG -g 3080Ti \
      zsh WordlenHintedUnitTranslation/scripts/run_battleship.sh

# ---------------------
Original_dataset_path=../AudioWords/data/CoVoSTUnits \
Unbinarized_dataset_path=data/FairseqCoVoSTUnits \
Binarized_dataset_path=data/BinFairseqCoVoSTUnits \
TRAIN_SPLIT=train \
DEV_SPLIT=dev \
TEST_SPLIT=test \
TGTDIR=translation \
TGT=de \
LENDIR=wordlengths LEN=wordlen \
hrun -s -c 16 -m 32 \
zsh WordlenHintedUnitTranslation/scripts/preprocess_data.sh

DATAPATH=data/BinFairseqCoVoSTUnits \
  MAXTOKENS=$((40960 / 3 / 2)) UPDATE_FREQ=2 LR=2e-4 EPOCHS=200 \
  WANDBPROJ=CVST_ST LOG_FILE=ST__2eNeg4 SAVE_DIR=ST_try1st__2eNeg4 \
      hrun -c 16 -m 32 -GGG -g TITANRTX \
      zsh WordlenHintedUnitTranslation/scripts/run_battleship_bleu.sh

#################$$$$$$$$$$$$$$$$$$$$$$
DATAPATH=data/BinFairseqLibriUnits \
  MAXTOKENS=$((40960 / 4)) LR=2e-4 EPOCHS=200 \
  WANDBPROJ=LS_AEASR LOG_FILE=AEASR_Second__2eNeg4 SAVE_DIR=AEASR_try2nd__2eNeg4 \
      hrun -c 16 -m 32 -GGGG -g 3080Ti \
      zsh WordlenHintedUnitTranslation/scripts/run_battleship.sh \
      --finetune-from-model AE_try1st__2eNeg4/checkpoint_best.pt --fix-encoder
