# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR=$1

# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="data/t5x_proc"
T5X_DIR=${HOME}"/t5x"  # directory where the T5X repo is cloned.

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="word_segmentation/t5x/configs/pretrain_mt5.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\"word_segmentation_mt5_pretrain\" \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/mt5_base/checkpoint_1000000\" \
  --gin.TRAIN_STEPS=1400000 \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --gin.trainer.Trainer.num_microbatches=2 &> segmenterv5-mt5.log

# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR=$2

# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="data/t5x_proc"
T5X_DIR=${HOME}"/t5x"  # directory where the T5X repo is cloned.

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="word_segmentation/t5x/configs/finetune_mt5.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\"word_segmentation_mt5_finetune\" \
  --gin.INITIAL_CHECKPOINT_PATH=\"$1/checkpoint_1400000\" \
  --gin.TRAIN_STEPS=1500000 \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --gin.trainer.Trainer.num_microbatches=2 &> segmenterv5-mt5-ft.log

# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR=$3

# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="data/t5x_proc"
T5X_DIR=${HOME}"/t5x"  # directory where the T5X repo is cloned.

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="word_segmentation/t5x/configs/finetune_mt5.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\"word_segmentation_mt5_finetune\" \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/mt5_base/checkpoint_1000000\" \
  --gin.TRAIN_STEPS=1100000 \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --gin.trainer.Trainer.num_microbatches=2 &> segmenterv5-mt5-ft-direct.log
