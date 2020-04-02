julia prep_ada.jl \
--defs ./data/new/train.json ./data/new/valid.json ./data/new/test.json \
--save ./data/new/processed/train_ada.npy ./data/new/processed/val_ada.npy ./data/new/processed/test_ada.npy \
--ada ./data/ada/adagram-model-wiki+ukwac+gigaword+news-lemma
