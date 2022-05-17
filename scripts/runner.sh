#!sh
fairseq-train data/BinFairseqLibriUnits --user-dir wordlen_hinted --task wordlen_translation \
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
