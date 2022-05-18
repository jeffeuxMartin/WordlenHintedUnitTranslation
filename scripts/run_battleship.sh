

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

fairseq-train data/BinFairseqLibriUnits \
    --lr 2e-3 \
    --max-tokens 4096 \
    \
   --user-dir WordlenHintedUnitTranslation/src \
   --task wordlen_translation \
   --arch iwslt_wordlen_transformer --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    \
    ` # learning ` \
    --clip-norm 1.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    \
    ` # regularization ` \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    \
    ` # training ` \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    \
    ` # logging ` \
    --wandb-project ok_fairseq \
    --log-file mynewlogs \
    --save-dir mynewsaveddir \
    \
    ` # saving ` \
    --keep-best-checkpoints 5 \
    --keep-last-epochs 5 \
    ` # --best-checkpoint-metric wer ` \
    ` # --maximize-best-checkpoibnt-metric ` \
    \
    --fp16 \
    \
    --scoring wer \
    --find-unused-parameters \
    --max-epoch 200 \
    # --maximize-best-checkpoint-metric --best-checkpoint-metric bleu --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples
    # ` # evaluation ` \
    # --eval-bleu \
    # --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    # --eval-bleu-detok moses \
    # --eval-bleu-remove-bpe \
    # --eval-bleu-print-samples \
    # \

# TODO: length weight (w)
# TODO: min length
# TODO: input alpha?
# TODO: colllen (用 N not Wordlen)
# TODO: ASR lower?
# TODO: pretrained?


# region ~~~~ NOTE ~~~~~~~~~~~ #
# lengthfilename (somehow fixed)
# use_self (fixed)
# args.use_self (fixed)
# paddingright (fixed)
# return_all (fixed)
# endregion ~ NOTE ~~~~~~~~~~~ #

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
