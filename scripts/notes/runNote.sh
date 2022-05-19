# silly .txt !!!
TGT=en LENDIR=wordlengths LEN=wordlen \ 
  DATAPATH=data/BinFairseqLibriUnits \
  MAXTOKENS=4096 LR=2e-4 EPOCHS=200 \
  WANDBPROJ=LS_ASR LOG_FILE=ASR_first SAVE_DIR=ASR_try1st \
      hrun -s -c 16 -m 32 \
      zsh WordlenHintedUnitTranslation/scripts/preprocess_data.sh

# # # # # # # # # # # # # # # # # # # # # # # # # # #
DATAPATH=data/BinFairseqLibriUnits \
  MAXTOKENS=10240 LR=2e-4 EPOCHS=200 \
  WANDBPROJ=LS_ASR LOG_FILE=ASR_Second__2eNeg4 SAVE_DIR=ASR_try2nd__2eNeg4 \
      hrun -s -c 16 -m 32 -GGGG -g 3080Ti \
      zsh WordlenHintedUnitTranslation/scripts/run_battleship.sh


DATAPATH=data/BinFairseqLibriUnits \
  MAXTOKENS=20480 LR=2e-3 EPOCHS=200 \
  WANDBPROJ=LS_ASR LOG_FILE=ASR_Second__2eNeg3 SAVE_DIR=ASR_try2nd__2eNeg3 \
      hrun -s -c 16 -m 32 -GG -g 3090 \
      zsh WordlenHintedUnitTranslation/scripts/run_battleship.sh

