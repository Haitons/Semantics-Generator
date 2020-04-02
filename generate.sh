CUDA_VISIBLE_DEVICES=7 python generate.py \
--params \
./checkpoints/params.json \
--ckpt \
./checkpoints/a.pkl \
--length 15 \
--strategy Multinomial \
--tau 0.05 
