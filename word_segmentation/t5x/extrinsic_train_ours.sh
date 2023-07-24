# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR=$2

# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="data/t5x_proc"
T5X_DIR=${HOME}"/t5x"  # directory where the T5X repo is cloned.

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="word_segmentation/t5x/configs/extrinsic_pretrain_our_mt5.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR} &> extrinsic_mt5_ours_v3.log
