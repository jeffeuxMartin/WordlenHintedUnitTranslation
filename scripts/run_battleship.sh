

cd /home/jeffeuxmartin/FairseqAudioWords

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

fairseq-train data/BinFairseqLibriUnits \
   --user-dir WordlenHintedUnitTranslation/src \
   --task wordlen_translation \
   --arch iwslt_wordlen_transformer --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    \
    ` # learning ` \
    --lr 2e-4 \
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
    --max-tokens 10240 \
    \
    ` # logging ` \
    --wandb-project ok_fairseq \
    --log-file mylogs \
    --save-dir mysaveddir \
    \
    ` # saving ` \
    --keep-best-checkpoints 5 \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    \
    ` # evaluation ` \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    \
    --fp16

