#!zsh

fairseq-train \
    ` # =========== ` \
    ` # dataset ` \
    $DATAPATH \
    ` # =========== ` \
    ` # learning ` \
    --lr $LR \
    --max-epoch $EPOCHS \
    --max-tokens $MAXTOKENS \
    ` # =========== ` \
    ` # =========== ` \
    ` # =========== ` \
    --user-dir WordlenHintedUnitTranslation/src \
    --task wordlen_translation \
    --arch iwslt_wordlen_transformer \
    --criterion aug_label_smoothed_cross_entropy \
    ` # =========== ` \
    ` # training ` \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 1.0 \
    \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    \
    `     # regularization ` \
    --label-smoothing 0.1 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    \
    ` # acceleration ` \
    --fp16 \
    --find-unused-parameters \
    \
    \
    ` # logging ` \
    --wandb-project $WANDBPROJ \
    --log-file $LOG_FILE \
    --save-dir $SAVE_DIR \
    \
    ` # saving ` \
    --keep-best-checkpoints 5 \
    --keep-last-epochs 5 \
    --scoring bleu \
    \
    ` # evaluation ` \
    --maximize-best-checkpoint-metric \
    --best-checkpoint-metric bleu \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    \
    
# --update-freq 2
