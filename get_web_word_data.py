import json
from dataclasses import dataclass
import random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

import pandas as pd
from transformers import HfArgumentParser
from tqdm.auto import tqdm

from word_segmentation.utils import LANGUAGES, LabelArgs, corrupt_text


@dataclass
class Args:
    output: str = "data/pretrain"
    valid_percent: int = 0.001
    wiktionary_valid_path: str = "data/wiktionary/valid.parquet"


if __name__ == "__main__":
    tqdm.pandas()

    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    wiktionary_valid_df = pd.read_parquet(args.wiktionary_valid_path)
    wiktionary_valid_words = set(wiktionary_valid_df["word"].str.lower().tolist())

    args.output = Path(args.output)
    args.output.mkdir(exist_ok=True, parents=True)

    train_data = set()
    valid_data = set()

    words_stripped = {}
    dash_word_counts = {lang: 0 for lang in LANGUAGES}
    non_dash_word_counts = {lang: 0 for lang in LANGUAGES}

    n_dropped_through_ratio = 0

    for lang in tqdm(LANGUAGES):
        lang_data = json.load(open(f"mc4_words/data/{lang}_final_dash.json"))
        lang_data_non_dash = json.load(open(f"mc4_words/data/{lang}_final_non_dash.json"))

        non_dash_freqs = defaultdict(lambda: 0)
        for key, value in lang_data_non_dash.items():
            non_dash_freqs[key.lower()] += value 

        for word, freq in lang_data.items():
            word = word.replace("‐", "-")
            if "-" not in word:
                continue

            if min(len(w) for w in word.split("-")) < 2:
                continue

            stripped_word = (word.replace("-", "").lower(), lang)
            ratio = freq / non_dash_freqs.get(stripped_word[0], 1)

            if ratio < np.exp(-6):
                n_dropped_through_ratio += 1
                print(f"Dropping {word} because of ratio ({ratio:.5f}). ({lang})")
                continue

            if stripped_word in words_stripped:
                lookup_freq, lookup_word = words_stripped[stripped_word]

                if lookup_freq < freq:
                    print(f"Dropping {lookup_word} for {word}. ({lang})")
                    words_stripped[stripped_word] = (freq, word)
                    try:
                        train_data.remove((lookup_word, lang))
                    except KeyError:
                        valid_data.remove((lookup_word, lang))
            else:
                words_stripped[stripped_word] = (freq, word)
                dash_word_counts[lang] += 1

            if random.random() < args.valid_percent or word.lower() in wiktionary_valid_words:
                valid_data.add((word, lang))
            else:
                train_data.add((word, lang))

    for lang in tqdm(LANGUAGES):
        lang_data_negatives = json.load(
            open(f"mc4_words/data/{lang}_final_non_dash.json")
        )
        for word, _ in Counter(lang_data_negatives).most_common():
            if (word.lower(), lang) in words_stripped:
                continue

            assert "-" not in word

            non_dash_word_counts[lang] += 1

            if non_dash_word_counts[lang] > dash_word_counts[lang]:
                break

            if random.random() < args.valid_percent or word.lower() in wiktionary_valid_words:
                valid_data.add((word, lang))
            else:
                train_data.add((word, lang))

    train_df = pd.DataFrame(train_data, columns=["word", "lang"])
    valid_df = pd.DataFrame(valid_data, columns=["word", "lang"])

    train_label_args = LabelArgs(dash_continuity_prob=0.999)
    valid_label_args = LabelArgs(dash_continuity_prob=1.0)

    train_df["input"] = train_df["word"].progress_apply(
        lambda x: corrupt_text(x, train_label_args)[0]
    )
    valid_df["input"] = valid_df["word"].progress_apply(
        lambda x: corrupt_text(x, valid_label_args)[0]
    )

    train_df[["input", "word", "lang"]].to_csv(
        args.output / "train.tsv", sep="\t", index=False, header=False
    )
    valid_df[["input", "word", "lang"]].to_csv(
        args.output / "valid.tsv", sep="\t", index=False, header=False
    )
