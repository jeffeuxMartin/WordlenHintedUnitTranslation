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

cd /home/jeffeuxmartin/FairseqAudioWords


# ` # data/BinDummyUnits `
# ` # 4096 `
# ` # 5e-4 `
# ` # 200 `

DATAPATH=$DATAPATH \
\
MAXTOKENS=$MAXTOKENS \
LR=$LR \
EPOCHS=$EPOCHS \
\
WANDBPROJ=$WANDBPROJ \
LOG_FILE=$LOG_FILE \
SAVE_DIR=$SAVE_DIR \
zsh WordlenHintedUnitTranslation/scripts/core_battleship.sh $@

# TODO: length weight (w)
# TODO: min length
# TODO: input alpha?
# TODO: colllen (用 N not Wordlen)
# TODO: ASR lower?
# TODO: pretrained?

# XXX: silly .txt for English!
# region ~~~ TODO ~~~~~~~~~~~ #
#   用 wordlen by space!
#   AE --> ASR --> ST
# * len weight / colllen 用 N / min length
#   抓到 alignment!
# - input alphas
# @ fairseq len / wordlen 自由切換
# @ pretrain BART?
# @ HuggingFace
# v lowercase
# endregion~ TODO ~~~~~~~~~~~ #


# region ~~~~ NOTE ~~~~~~~~~~~ #
# lengthfilename (somehow fixed)
# use_self (fixed)
# args.use_self (fixed)
# paddingright (fixed)
# return_all (fixed)

# # # ＣＣＣ：
# # # 可以調阿==
# # # 可以覆寫load weight的函式阿==
# endregion ~ NOTE ~~~~~~~~~~~ #

# region ~~~~~~~~~~~~~~~~~~~~~~~~~~ PLAN FIXME XXX ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# * [x]    3. collunit_CIF___ASR 
# * [x]    5. collunit_CIF_AE    
# * [x]    4. collunit_CIF___ST  
#   * [o]  6. collunit_CIF_AEASR 
#   * [v]  7. collunit_CIF_AEST  
# ----------------------
# # Chap3
# * [x]    1. collunit_Tfm___ASR
# * [x]    2. collunit_Tfm___ST
# * [o]    8. fullunit_CIF_AE
#   * [ ]  9. fullunit_CIF_AEASR
#   * [ ] 10. fullunit_CIF_AEST
# * [v]   11. collunit_Tfm_AE
#   * [v] 12. collunit_Tfm_AEASR
#   * [v] 13. collunit_Tfm_AEST
# (CIF   --> others2s)
# (ST    --> ASRST     --> IC)
# (AE    --> AEinit)
# (BART  --> scratrch)
# (end   --> front)
# (unit  --> feat)

# # Chap4
# * [@]  1. collunit_CIF_AE$selfL   
# * [@]  4. collunit_CIF_AE$minL     
# * [@]  7. collunit_CIF_AE$1L     
# * [@] 10. collunit_CIF_AE$FullL     

# * [ ]  2. collunit_CIF_AE$selfL_ASR 
# * [ ]  3. collunit_CIF_AE$selfL_ST  
# * [ ]  5. collunit_CIF_AE$minL_ASR 
# * [ ]  6. collunit_CIF_AE$minL_ST  
# * [ ]  8. collunit_CIF_AE$1L_ASR 
# * [ ]  9. collunit_CIF_AE$1L_ST  
# * [ ] 11. collunit_CIF_AE$FullL_ASR 
# * [ ] 12. collunit_CIF_AE$FullL_ST  
# (selfL --> supL)

# endregion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PLAN FIXME XXX ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# region ~~~~~~~~~~~~~~~~~~~~~~~~~~ PLAN FIXME XXX ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# [scratrch]|weightlen_5strategy|weight3len|coll3len
# 
# * [ ] 1. collunit CIF AE
# * [ ]         // 2. collunit CIF (load AE init) ASR
# * [ ] 3. collunit CIF (load AE fix) ASR
# * [x] 4. collunit CIF ASR
# * [ ] 5. fullunit CIF AE
# * [ ]         // 6. fullunit CIF (load AE init) ASR
# * [ ] 7. fullunit CIF (load AE fix) ASR
# * [o] 8. fullunit CIF ASR
# * [x] 4. collunit noCIF ASR
# * [o] 8. fullunit noCIF ASR
# * [ ]     1. collunit others2s AE
# * [ ]             // 2. collunit others2s (load AE init) ASR
# * [ ]     3. collunit others2s (load AE fix) ASR
# * [ ]     4. collunit others2s ASR
# * [ ]     5. fullunit others2s AE
# * [ ]             // 6. fullunit others2s (load AE init) ASR
# * [ ]     7. fullunit others2s (load AE fix) ASR
# * [ ]     8. fullunit others2s ASR
# 
# 
# 
# * [ ]         // 2. collunit CIF (load AE init) ST
# * [ ] 3. collunit CIF (load AE fix) ST
# * [x] 4. collunit CIF ST
# * [ ]         // 6. fullunit CIF (load AE init) ST
# * [ ] 7. fullunit CIF (load AE fix) ST
# * [v] 8. fullunit CIF ST
# * [x] 4. collunit noCIF ST
# * [v] 8. fullunit noCIF ST
# * [ ]             // 2. collunit others2s (load AE init) ST
# * [ ]     3. collunit others2s (load AE fix) ST
# * [ ]     4. collunit others2s ST
# * [ ]             // 6. fullunit others2s (load AE init) ST
# * [ ]     7. fullunit others2s (load AE fix) ST
# * [ ]     8. fullunit others2s ST
# * [ ]         // 2. collunit CIF (load ASR init) ST
# * [ ] 3. collunit CIF (load ASR fix) ST
# * [ ]         // 6. fullunit CIF (load ASR init) ST
# * [ ] 7. fullunit CIF (load ASR fix) ST
# * [ ]             // 2. collunit others2s (load ASR init) ST
# * [ ]     3. collunit others2s (load ASR fix) ST
# * [ ]             // 6. fullunit others2s (load ASR init) ST
# * [ ]     7. fullunit others2s (load ASR fix) ST
# 
# 
# 
# * [ ]             // 2. collunit CIF (load AE init) IC
# * [ ]     3. collunit CIF (load AE fix) IC
# * [ ]     4. collunit CIF IC
# * [ ]             // 6. fullunit CIF (load AE init) IC
# * [ ]     7. fullunit CIF (load AE fix) IC
# * [ ]     8. fullunit CIF IC
# * [ ]     4. collunit noCIF IC
# * [ ]     8. fullunit noCIF IC
# * [ ]                 // 2. collunit others2s (load AE init) IC
# * [ ]         3. collunit others2s (load AE fix) IC
# * [ ]         4. collunit others2s IC
# * [ ]                 // 6. fullunit others2s (load AE init) IC
# * [ ]         7. fullunit others2s (load AE fix) IC
# * [ ]         8. fullunit others2s IC
# * [ ]             // 2. collunit CIF (load ASR init) IC
# * [ ]     3. collunit CIF (load ASR fix) IC
# * [ ]             // 6. fullunit CIF (load ASR init) IC
# * [ ]     7. fullunit CIF (load ASR fix) IC
# * [ ]                 // 2. collunit others2s (load ASR init) IC
# * [ ]         3. collunit others2s (load ASR fix) IC
# * [ ]                 // 6. fullunit others2s (load ASR init) IC
# * [ ]         7. fullunit others2s (load ASR fix) IC
# endregion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PLAN FIXME XXX ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
