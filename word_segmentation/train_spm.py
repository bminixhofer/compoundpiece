from dataclasses import dataclass
from transformers import HfArgumentParser
import tensorflow as tf
from collections import Counter
from tqdm.auto import tqdm
import regex as re
import os
from pathlib import Path
import sentencepiece as spm
import subprocess
import json
import pickle

from word_segmentation.utils import LANGUAGES, segment
from get_wiktionary_data import WiktextractCompoundExtractor

FEATURE_DESCRIPTION = {
    "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "lang": tf.io.FixedLenFeature([], tf.string, default_value=""),
}

PUNCT_REGEX = re.compile("(\p{P}|\p{S})")

def words_to_constituents(words, output_dir):
    words_file = output_dir / "words.tsv"

    open(words_file, "w").writelines([word.strip() + "\n" for word in words])

    os.environ["T5X_TEST_TSV"] = str(words_file)

    infer_output_dir = output_dir / "infer"
    subprocess.run(
        [
            "python3",
            f"{os.environ['T5X_DIR']}/t5x/infer.py",
            "--gin_file=word_segmentation/t5x/configs/infer_byt5.gin",
            f'--gin.CHECKPOINT_PATH="{args.decompound_checkpoint_path}"',
            f'--gin.INFER_OUTPUT_DIR="{infer_output_dir}"',
        ],
        check=True,
    )

    all_constituents = []

    for line, word in tqdm(
        zip(
            open(
                infer_output_dir
                / "word_segmentation_byte_infer-predict.jsonl-00000-of-00001"
            ),
            words,
        ),
        total=len(words),
        desc="Loading predictions...",
    ):
        parts = json.loads(line)["prediction"].split("-")

        if len(parts) > 1:
            reconstructed = WiktextractCompoundExtractor.levenshtein_split(word, parts)
            if reconstructed is not None:
                constituents, _ = reconstructed
            else:
                constituents = [word]
        else:
            constituents = [word]

        all_constituents.append(constituents)

    return all_constituents


@dataclass
class Args:
    output: str
    decompound_checkpoint_path: str # path to t5x checkpoint
    train_file: str = "data/raw_text/train.tfrecord"
    max_length: int = 32
    batch_size: int = 128 * 8
    resolution: int = 16
    vocab_size: int = 250_112
    per_lang_vocab_size: int = 32768
    num_proc: int = 64
    decompound: bool = False


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    print(args)

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    dataset = tf.data.TFRecordDataset(
        [args.train_file], compression_type="GZIP", num_parallel_reads=args.num_proc
    )
    word_counters = {lang: Counter() for lang in LANGUAGES}

    dataset = dataset.map(
        lambda x: tf.io.parse_single_example(x, FEATURE_DESCRIPTION),
        num_parallel_calls=args.num_proc,
    )

    for sample in tqdm(dataset):
        lang = sample["lang"].numpy().decode("utf-8")
        text = sample["text"].numpy().decode("utf-8")

        word_counters[lang].update([prefix + word for prefix, word in segment(text)])

    counter_dir = output_dir / "word_counters"
    counter_dir.mkdir(exist_ok=True, parents=True)
    for lang, counter in word_counters:
        pickle.dump(counter, open(counter_dir / f"{lang}.pkl", "wb"))

    aggregated_counter = Counter()
    for counter in word_counters.values():
        aggregated_counter += counter

    train_words = {}
    predict_words = []

    for key, value in tqdm(aggregated_counter.most_common()):
        if value < args.resolution:
            continue

        if len(key.encode("utf-8")) >= args.max_length:
            continue

        if (
            "-" in key
            or "‐" in key
            or all(c.isdigit() or c.isspace() or PUNCT_REGEX.match(c) for c in key)
        ):
            train_words[key] = [[key], value // args.resolution]
        else:
            train_words[key] = [None, value // args.resolution]
            predict_words.append(key)

    if args.decompound:
        all_constituents = words_to_constituents(predict_words, output_dir)

        for word, constituents in zip(predict_words, all_constituents):
            assert train_words[word][0] is None
            train_words[word][0] = constituents
    else:
        for word in train_words.keys():
            if train_words[word][0] is None:
                train_words[word][0] = [word]

    json.dump(train_words, open(output_dir / "train_words.json", "w"), indent=4)

    def all_sentences():
        for word, (constituents, count) in train_words.items():
            for i, constituent in enumerate(constituents):
                for _ in range(count):
                    yield constituent

    spm.SentencePieceTrainer.train(
        sentence_iterator=all_sentences(),
        model_prefix=str(output_dir / "tokenizer"),
        vocab_size=args.vocab_size,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        train_extremely_large_corpus=True,
    )

    lang_out_dir = output_dir / "monolingual"
    lang_out_dir.mkdir(exist_ok=True, parents=True)

    for lang in tqdm(LANGUAGES):
        low_res_lang_counter = Counter()
        for key, count in word_counters[lang].items():
            if key not in train_words:
                continue

            if count >= args.resolution:
                low_res_lang_counter[key] = count // args.resolution

        def sentences():
            for word, count in low_res_lang_counter.items():
                constituents, _ = train_words[word]

                for i, constituent in enumerate(constituents):
                    for _ in range(count):
                        yield constituent

        spm.SentencePieceTrainer.train(
            sentence_iterator=sentences(),
            model_prefix=str(lang_out_dir / lang),
            vocab_size=args.per_lang_vocab_size,
            hard_vocab_limit=False,
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            train_extremely_large_corpus=True,
        )
