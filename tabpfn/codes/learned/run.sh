python scripts/generate_prior_data.py \
  --prior_type mlp_scm \
  --num_batches 5000 \
  --batch_size 512 \
  --n_jobs 12 \
  --max_seq_len 256 \
  --max_features 20 \
  --max_classes 10 \
  --seed 42 \
  --save_path data/scm_5k.h5

# Tree (約8時間, 256万データセット)
python scripts/generate_prior_data.py \
  --prior_type tree_scm \
  --num_batches 5000 \
  --batch_size 512 \
  --n_jobs 12 \
  --max_seq_len 256 \
  --max_features 20 \
  --max_classes 10 \
  --seed 43 \
  --save_path data/tree_5k.h5

