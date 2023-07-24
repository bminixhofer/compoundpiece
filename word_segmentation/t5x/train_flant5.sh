# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR=$1 # stage 1

# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="data/t5x_proc"
T5X_DIR=${HOME}"/t5x"  # directory where the T5X repo is cloned.

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="word_segmentation/t5x/configs/pretrain_t5.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\"word_segmentation_t5_pretrain\" \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/flan_t5_base/checkpoint_1184000\" \
  --gin.TRAIN_STEPS=1584000 \
  --tfds_data_dir=${TFDS_DATA_DIR} &> segmenterv5-flant5.log

# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR=$2 # stage 2 from stage 1

# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="data/t5x_proc"
T5X_DIR=${HOME}"/t5x"  # directory where the T5X repo is cloned.

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="word_segmentation/t5x/configs/finetune_t5.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\"word_segmentation_t5_finetune\" \
  --gin.INITIAL_CHECKPOINT_PATH=\"$1/checkpoint_1584000\" \
  --gin.TRAIN_STEPS=1684000 \
  --tfds_data_dir=${TFDS_DATA_DIR} &> segmenterv5-flant5-ft.log

# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR=$3 # stage 2 only

# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="data/t5x_proc"
T5X_DIR=${HOME}"/t5x"  # directory where the T5X repo is cloned.

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="word_segmentation/t5x/configs/finetune_t5.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\"word_segmentation_t5_finetune\" \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/flan_t5_base/checkpoint_1184000\" \
  --gin.TRAIN_STEPS=1284000 \
  --tfds_data_dir=${TFDS_DATA_DIR} &> segmenterv5-flant5-ft-direct.log
