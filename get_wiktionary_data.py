import itertools
import math
from pathlib import Path
import Levenshtein
from tqdm.auto import tqdm
import json
from dataclasses import dataclass
from transformers import HfArgumentParser, AutoTokenizer
import numpy as np
import pandas as pd
from pathlib import Path

tqdm.pandas()

from word_segmentation.utils import (
    LANGUAGES,
    LabelArgs,
    corrupt_text,
    levenshtein_label,
)

WIKTEXTRACT_DATA_PATH = "data/external/raw-wiktextract-data.json"
GERMANET_DATA_PATH = (
    "data/external/split_compounds_from_GermaNet17.0_modified-2022-06-28.txt"
)


@dataclass
class Args:
    output_dir: str = "data/gold"


class WiktextractCompoundExtractor:
    @classmethod
    def get_offsets_by_sum(cls, constituents, s, target_delta):
        return (
            x
            for x in itertools.product(
                *(range(max(-s, -len(c)), min(s + 1, len(c) + 1)) for c in constituents)
            )
            if sum((abs(i) for i in x)) == s and sum(x) == target_delta
        )

    @classmethod
    def offsets_to_constituents(cls, word, constituents, offsets):
        i = 0
        for c, o in zip(constituents, offsets):
            yield word[i : i + len(c) + o]
            i += len(c) + o

    @classmethod
    def levenshtein_split(cls, word, constituents, prefer_prefix=False):
        assert len(constituents) > 1

        best_distance = math.inf
        best_offsets = None
        current_min_distance = 0

        target_delta = len(word) - sum(len(c) for c in constituents)

        while current_min_distance <= best_distance:
            all_offsets = list(
                cls.get_offsets_by_sum(constituents, current_min_distance, target_delta)
            )

            for offsets in all_offsets:
                distance = 0

                for w, c in zip(
                    cls.offsets_to_constituents(word, constituents, offsets),
                    constituents,
                ):
                    current_distance = Levenshtein.distance(w.lower(), c.lower())
                    distance += current_distance

                if distance < best_distance:
                    best_distance = distance
                    best_offsets = [offsets]
                elif distance == best_distance:
                    best_offsets.append(offsets)

            current_min_distance += 1
            if current_min_distance >= len(word):
                break

        if best_offsets is None:
            return None

        best_candidate = list(
            cls.offsets_to_constituents(
                word, constituents, best_offsets[0 if prefer_prefix else -1]
            )
        )

        if any(len(c) == 0 for c in best_candidate):
            return None

        return best_candidate, best_distance

    @classmethod
    def is_compound(cls, word):
        categories = []
        if "categories" in word:
            categories.extend(word["categories"])
        if "senses" in word:
            for sense in word["senses"]:
                if "categories" in sense:
                    categories.extend(sense["categories"])

        return any("compound" in category.lower().split() for category in categories)

    @classmethod
    def get_compound_constituents(cls, lang_code, word):
        text = word["word"]
        best_distance = math.inf
        best_candidate = None
        best_raw_constituents = None

        for template in word["etymology_templates"]:
            constituents = []

            for i, arg in template["args"].items():
                if arg == lang_code:
                    continue

                try:
                    i = int(i)
                except ValueError:
                    continue

                arg = arg.strip()

                # connective, do not include
                # also do not include constituents of length one, will mostly (fully?) be connectives
                if arg.startswith("-") or arg.endswith("-") or len(arg) == 1:
                    continue

                if len(arg) > 0:
                    constituents.append(arg)

            if len(constituents) < 2:
                continue

            if any("-" in c for c in constituents):
                continue

            out = cls.levenshtein_split(text, constituents)

            if out is None:
                continue

            candidate, distance = out

            # sometimes language prefix is used
            constituents = [
                constituent
                for constituent, c in zip(constituents, candidate)
                if len(c) > 0
            ]
            candidate = [c for c in candidate if len(c) > 0]

            if distance < best_distance:
                best_distance = distance
                best_candidate = candidate
                best_raw_constituents = constituents

        if best_distance == math.inf:
            return None

        return best_distance, best_candidate, best_raw_constituents

    @classmethod
    def extract(cls, wiktextract_data_path, languages_to_include):
        raw_compound_words = {lang_code: [] for lang_code in languages_to_include}
        non_compound_words = {lang_code: [] for lang_code in languages_to_include}

        # total computed from `wc -l ...`, not necessarily accurate
        for line in tqdm(open(wiktextract_data_path), total=8395231):
            data = json.loads(line)

            if "sounds" in data:
                del data["sounds"]

            if data.get("lang_code") in languages_to_include:
                if (
                    any(c.isspace() for c in data["word"])
                    or "-" in data["word"]
                    or all(c.isupper() for c in data["word"])
                ):
                    continue

                if cls.is_compound(data):
                    raw_compound_words[data["lang_code"]].append(data)
                else:
                    non_compound_words[data["lang_code"]].append(data["word"])

        compounds = {}

        for lang_code, lang_words in tqdm(raw_compound_words.items()):
            compounds_without_info = set()
            proc_words = {}

            for word in lang_words:
                assert cls.is_compound(word)
                if "etymology_templates" not in word:
                    compounds_without_info.add(word["word"].lower())
                    continue

                out = cls.get_compound_constituents(lang_code, word)

                if out is not None:
                    score, constituents, raw_constituents = out

                    if len(constituents) == 1:
                        continue

                    if any(("#" in c or " " in c) for c in raw_constituents):
                        continue

                    proc_words[word["word"]] = (constituents, raw_constituents)

            # wiktionary compounds often have only one split, e.g. https://en.wiktionary.org/wiki/Abendsonnenschein#German
            # so recursively split compound constituents into their smallest parts here
            decompounded_proc_words = {}

            def decompound_recurse(parent, constituents, raw_constituents):
                out = []
                raw_out = []

                for c, x in zip(constituents, raw_constituents):
                    if x in compounds_without_info:
                        return None
                    if x in proc_words and x != parent:
                        v = decompound_recurse(x, *proc_words[x])

                        if v is None:
                            return None

                        split_out = cls.levenshtein_split(c, v[0])

                        if split_out is None:
                            return None

                        out.extend(split_out[0])
                        raw_out.extend(v[1])
                    else:
                        out.append(c)
                        raw_out.append(x)

                return out, raw_out

            for key, value in proc_words.items():
                parts = decompound_recurse(key, *value)

                if parts is not None:
                    decompounded_proc_words[key] = (key, *parts)

            positives = sorted(decompounded_proc_words.values(), key=lambda x: x[0])

            negatives = [
                constituent
                for (_, _, raw_constituents) in positives
                for constituent in raw_constituents
            ]

            compounds[lang_code] = {
                "positives": positives,
                "negatives": negatives,
                "unknown": non_compound_words[lang_code],
            }

        return compounds


if __name__ == "__main__":
    (args,) = HfArgumentParser(
        [
            Args,
        ]
    ).parse_args_into_dataclasses()

    np.random.seed(1234)

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True, parents=True)

    wiktionary_compounds = WiktextractCompoundExtractor.extract(
        WIKTEXTRACT_DATA_PATH, LANGUAGES
    )

    lines = open(GERMANET_DATA_PATH).readlines()
    germanet_compounds = []
    for i, line in tqdm(enumerate(lines), total=len(lines)):
        if i < 2:
            continue

        word, tail, head = line.strip().split("\t")

        if "-" in word:
            # consistent with SECOS
            continue

        # e.g. 'Einkaufskonditionen  Einkauf Kondition' needs this treatment instead
        # instead of just taking the length of the head
        try:
            head_start = word.lower().index(head.lower())
        except ValueError:
            print(f"WARNING: {(word, tail, head)} processing failed.")

        germanet_compounds.append((word, (word[:head_start], word[head_start:])))

    rows = []
    for lang, compounds in wiktionary_compounds.items():
        for word, constituents, raw_constituents in compounds["positives"]:
            rows.append(
                ("-".join(constituents), lang, "positive", "-".join(raw_constituents))
            )

        for word in compounds["negatives"]:
            rows.append((word, lang, "negative", ""))

        for word in compounds["unknown"]:
            rows.append((word, lang, "unknown", ""))

    df = (
        pd.DataFrame(
            rows,
            columns=["word", "lang", "type", "norm"],
        )
        .sample(frac=1)
        .reset_index(drop=True)
    )

    df = pd.concat([df[df["type"] == x] for x in ["positive", "negative", "unknown"]])
    df = df[~df.duplicated("word")].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

    def get_distance(row):
        if len(row["norm"]) == 0:
            return 0

        return len(
            levenshtein_label(
                *corrupt_text(row["word"], LabelArgs(dash_continuity_prob=1.0)),
                row["norm"],
                tokenizer,
            )[1]
        )

    n_edits = df.progress_apply(
        get_distance,
        axis=1,
    )
    df = df[(n_edits < df["word"].apply(lambda x: len(x.encode("utf-8"))))]

    df_without_unknowns = df[df["type"] != "unknown"]

    eval_lang_counts = df_without_unknowns["lang"].value_counts()
    eval_sizes = eval_lang_counts.apply(lambda x: min(int(x * 0.5), 1_000)).to_dict()

    valid_indices = []
    for lang, size in eval_sizes.items():
        valid_indices.extend(
            df_without_unknowns[df_without_unknowns["lang"] == lang].sample(size).index
        )

    train_df = df.drop(valid_indices)
    valid_df = df.loc[valid_indices]

    german_wiktionary_words = set(
        [x[0] for x in wiktionary_compounds["de"]["positives"]]
    )

    test_df = pd.DataFrame(
        [
            ("-".join(x[1]), "positive", "de")
            for x in germanet_compounds
            if x[0] not in german_wiktionary_words
        ],
        columns=["word", "type", "lang"],
    )

    train_df.to_parquet(args.output_dir / "train.parquet")
    valid_df.to_parquet(args.output_dir / "valid.parquet")
    test_df.to_parquet(args.output_dir / "germanet.parquet")
