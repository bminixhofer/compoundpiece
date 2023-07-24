from datasets import load_dataset
import pandas as pd
from word_segmentation.utils import LANGUAGES
from tqdm.auto import tqdm
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
from transformers import HfArgumentParser

@dataclass
class Args:
    n_shards: int = 100
    n_train_pages: int = 500 * 1_000_000
    n_valid_pages: int = 100_000
    alpha: float = 0.2
    out_train_dir: str = "/mnt/disks/persist/data/raw_text/train/"
    out_valid_file: str = "/mnt/disks/persist/data/raw_text/valid.tfrecord"

if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    out_train_dir = Path(args.out_train_dir)
    out_valid_file = Path(args.out_valid_file)

    out_train_dir.mkdir(exist_ok=True, parents=True)
    out_valid_file.parent.mkdir(exist_ok=True, parents=True)

    mt5_metadata = pd.read_csv("mt5_metadata.csv").set_index("lang_code").loc[LANGUAGES]

    percent = mt5_metadata["pages"] ** args.alpha
    mt5_metadata["our_percent"] = percent / percent.sum()
    mt5_metadata["our_train_pages"] = (args.n_train_pages * mt5_metadata["our_percent"]).astype(
        int
    )
    mt5_metadata["our_valid_pages"] = (args.n_valid_pages * mt5_metadata["our_percent"]).astype(
        int
    )

    bar = tqdm(total=args.n_train_pages + args.n_valid_pages)


    def to_example(text, lang):
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "text": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[text.encode("utf-8")])
                    ),
                    "lang": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[lang.encode("utf-8")])
                    ),
                }
            )
        ).SerializeToString()

    train_writers = [
        tf.io.TFRecordWriter(
            str(out_train_dir / f"shard{i}.tfrecord"), options=tf.io.TFRecordOptions(compression_type="GZIP")
        ) for i in range(args.n_shards)
    ]
    valid_writer = tf.io.TFRecordWriter(
        str(out_valid_file), options=tf.io.TFRecordOptions(compression_type="GZIP")
    )

    current_shard = 0

    for lang in LANGUAGES:
        n_train_pages = mt5_metadata.loc[lang]["our_train_pages"]
        n_valid_pages = mt5_metadata.loc[lang]["our_valid_pages"]

        print(f"Downloading {lang} ({n_train_pages=}, {n_valid_pages=})")

        dset = load_dataset(
            "mc4/mc4.py",
            "iw" if lang == "he" else lang,
            streaming=True,
            split="train",
        )

        iterator = iter(dset)

        n_valid = 0
        n_train = 0

        for i in range(n_valid_pages):
            text = next(iterator)["text"].replace("\n", "").replace("\t", "")
            valid_writer.write(to_example(text, lang))

            bar.update(1)

        for i in range(n_train_pages):
            try:
                sample = next(iterator)
            except StopIteration:
                iterator = iter(dset)
                # skip valid
                for i in range(n_valid_pages):
                    next(iterator)

                sample = next(iterator)

            text = sample["text"].replace("\n", "").replace("\t", "")
            train_writers[current_shard].write(to_example(text, lang))

            current_shard += 1
            if current_shard >= len(train_writers):
                current_shard = 0

            bar.update(1)
