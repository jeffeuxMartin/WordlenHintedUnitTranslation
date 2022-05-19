#region Bad
# cp data/FairseqCoVoSTUnits/dev.de      data/FairseqCoVoSTUnits/dev.joint
# cp data/FairseqCoVoSTUnits/test.de     data/FairseqCoVoSTUnits/test.joint
# cp data/FairseqCoVoSTUnits/train.de    data/FairseqCoVoSTUnits/train.joint
# cp data/FairseqLibriAE/dev.collunit    data/FairseqLibriAE/dev.joint
# cp data/FairseqLibriAE/test.collunit   data/FairseqLibriAE/test.joint
# cp data/FairseqLibriAE/train.collunit  data/FairseqLibriAE/train.joint
# cp data/FairseqLibriUnits/dev.en       data/FairseqLibriUnits/dev.joint
# cp data/FairseqLibriUnits/test.en      data/FairseqLibriUnits/test.joint
# cp data/FairseqLibriUnits/train.en     data/FairseqLibriUnits/train.joint
#endregion Bad

#region FAKE
# fairseq-preprocess \
# -s collunit --srcdict data/dummy_collunit.dict \
# -t collunit \
# ` # --tgtdict data/BinFairseqLibriAE/dict.collunit.txt  ` \
# --trainpref data/FairseqLibriAE/train \   
# --validpref data/FairseqLibriAE/dev \
# --testpref data/FairseqLibriAE/test \
# --destdir data/BinFairseqLibriAE \
# --workers $(python -c 'import os; print(len(os.sched_getaffinity(0)))')

# TGTBIN=data/BinFairseqLibriUnits
# rm -rf $TGTBIN
# fairseq-preprocess \
# -s collunit --srcdict data/dummy_collunit.dict \
# -t en \
# ` # --tgtdict data/BinFairseqLibriAE/dict.collunit.txt ` \
# --trainpref data/FairseqLibriUnits/train,data/FairseqLibriAE/train,data/FairseqCoVoSTUnits/train \
# --validpref data/FairseqLibriUnits/dev,data/FairseqLibriAE/dev,data/FairseqCoVoSTUnits/dev \
# --testpref data/FairseqLibriUnits/test,data/FairseqLibriAE/test,data/FairseqCoVoSTUnits/test \
# --destdir $TGTBIN \
# --workers $(python -c 'import os; print(len(os.sched_getaffinity(0)))')












# fairseq-preprocess \
# -s collunit --srcdict data/dummy_collunit.dict \
# -t collunit \
# --tgtdict data/BinFairseqLibriAE/dict.collunit.txt \
# --trainpref data/FairseqLibriAE/train \   
# --validpref data/FairseqLibriAE/dev \
# --testpref data/FairseqLibriAE/test \
# --destdir data/BinFairseqLibriAE \
# --workers $(python -c 'import os; print(len(os.sched_getaffinity(0)))')
#endregion FAKE


cat data/FairseqLibriUnits/train.joint data/FairseqLibriAE/train.joint data/FairseqCoVoSTUnits/train.joint  > data/AllJoint.train.joint
cat data/FairseqLibriUnits/dev.joint data/FairseqLibriAE/dev.joint data/FairseqCoVoSTUnits/dev.joint  > data/AllJoint.dev.joint
cat data/FairseqLibriUnits/test.joint data/FairseqLibriAE/test.joint data/FairseqCoVoSTUnits/test.joint  > data/AllJoint.test.joint

fairseq-preprocess \
-s joint \
--trainpref data/AllJoint.train \
--validpref data/AllJoint.dev \
--testpref data/AllJoint.test \
--destdir data/BinAllDictOnly \
--dict-only \
--only-source \
--workers $(python -c 'import os; print(len(os.sched_getaffinity(0)))')

ls data/BinAllDictOnly/dict.joint.txt

# fairseq-preprocess \
# -s collunit \
# -t joint \
# --trainpref data/FairseqLibriUnits/train,data/FairseqLibriAE/train,data/FairseqCoVoSTUnits/train \
# --validpref data/FairseqLibriUnits/dev,data/FairseqLibriAE/dev,data/FairseqCoVoSTUnits/dev \
# --testpref data/FairseqLibriUnits/test,data/FairseqLibriAE/test,data/FairseqCoVoSTUnits/test \
# --destdir data/BinDictOnly \
# --dict-only \
# --joined-dictionary \
# --workers $(python -c 'import os; print(len(os.sched_getaffinity(0)))')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
fairseq-preprocess \
-s collunit --srcdict data/dummy_collunit.dict \
-t collunit --tgtdict data/BinAllDictOnly/dict.joint.txt \
--trainpref data/FairseqLibriAE/train \
--validpref data/FairseqLibriAE/dev \
--testpref data/FairseqLibriAE/test \
--destdir data/BinFairseqLibriAE \
--workers $(python -c 'import os; print(len(os.sched_getaffinity(0)))')

fairseq-preprocess \
-s collunit --srcdict data/dummy_collunit.dict \
-t en --tgtdict data/BinAllDictOnly/dict.joint.txt \
--trainpref data/FairseqLibriUnits/train \
--validpref data/FairseqLibriUnits/dev \
--testpref data/FairseqLibriUnits/test \
--destdir data/BinFairseqLibriUnits \
--workers $(python -c 'import os; print(len(os.sched_getaffinity(0)))')

fairseq-preprocess \
-s collunit --srcdict data/dummy_collunit.dict \
-t de --tgtdict data/BinAllDictOnly/dict.joint.txt \
--trainpref data/FairseqCoVoSTUnits/train \
--validpref data/FairseqCoVoSTUnits/dev \
--testpref data/FairseqCoVoSTUnits/test \
--destdir data/BinFairseqCoVoSTUnits \
--workers $(python -c 'import os; print(len(os.sched_getaffinity(0)))')
