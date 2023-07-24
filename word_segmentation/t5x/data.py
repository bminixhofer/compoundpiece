import pandas as pd
from word_segmentation.utils import corrupt_text, LabelArgs
from tqdm import tqdm

tqdm.pandas()

train_label_args = LabelArgs(dash_continuity_prob=0.999)
valid_label_args = LabelArgs(dash_continuity_prob=1.0)

train_df = pd.read_parquet("data/pretrain/train.parquet")
valid_df = pd.read_parquet("data/pretrain/valid.parquet")

train_df["input"] = train_df["word"].progress_apply(lambda x: corrupt_text(x, train_label_args)[0])
valid_df["input"] = valid_df["word"].progress_apply(lambda x: corrupt_text(x, valid_label_args)[0])

train_df[["input", "word"]].to_csv("data/t5x/pretrain_train.tsv", sep="\t", index=False, header=False)
valid_df[["input", "word"]].to_csv("data/t5x/pretrain_valid.tsv", sep="\t", index=False, header=False)
