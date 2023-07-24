from dataclasses import dataclass
import random
import unicodedata
import Levenshtein
import regex as re
import os
from pathlib import Path

import numpy as np


ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
LANGUAGES = [
    x.strip() for x in open(os.path.join(ROOT_DIR, "languages.txt")).readlines()
]
DASH_GROUP = (ord("-") + 3,)

@dataclass
class LabelArgs:
    dash_continuity_prob: float = 0.9


def corrupt_text(text, label_args):
    out_text = ""
    out_text_with_corrupted_case = ""

    i = 0
    while i < len(text):
        if text[i] == "-":
            assert i > 0 and i < len(text) - 1

            out_text_with_corrupted_case += "-"

            if random.random() < label_args.dash_continuity_prob:
                c_before = text[i - 1]
                c_after = text[i + 1]

                before_is_lowercase = c_before.lower() == c_before
                after_is_lowercase = c_after.lower() == c_after

                before_is_uppercase = c_before.upper() == c_before
                after_is_uppercase = c_after.upper() == c_after

                if before_is_lowercase and after_is_uppercase:
                    out_text += c_after.lower()
                    out_text_with_corrupted_case += c_after.lower()
                elif before_is_uppercase and after_is_lowercase:
                    out_text += c_after.upper()
                    out_text_with_corrupted_case += c_after.upper()
                else:
                    out_text += c_after
                    out_text_with_corrupted_case += c_after

                i += 2
            else:
                out_text += text[i + 1]
                out_text_with_corrupted_case += text[i + 1]
                i += 2
        else:
            out_text += text[i]
            out_text_with_corrupted_case += text[i]
            i += 1

    return out_text, out_text_with_corrupted_case

def apply_edits(input_ids, labels):
    out = []

    for i, label in enumerate(labels):
        assert ((label[2] != 0) + (label[1] != 0)) < 2

        if label[1] != 0:
            out.append(label[1])
        elif label[2] == 0:
            out.append(input_ids[i])

        if label[0] != 0:
            out.insert(len(out) - 1, label[0])

    return out


def byte_edit_ops(a, b):
    return Levenshtein.editops("".join(chr(i) for i in a), "".join(chr(i) for i in b))


def levenshtein_label(c_word, word, norm_word, tokenizer):
    norm_input_ids = tokenizer.encode(norm_word)
    segmented_input_ids = tokenizer.encode(word)
    input_ids = tokenizer.encode(c_word)

    all_labels = [np.zeros((len(input_ids), 3), dtype=np.uint8)]
    all_input_ids = [input_ids]

    for name, src_i, tgt_i in byte_edit_ops(input_ids, segmented_input_ids):
        byte = segmented_input_ids[tgt_i]

        if name == "insert":
            assert all_labels[-1][src_i, 0] == 0
            all_labels[-1][src_i, 0] = byte
        elif name == "replace":
            assert all_labels[-1][src_i, 1] == 0
            all_labels[-1][src_i, 1] = byte
        elif name == "delete":
            assert all_labels[-1][src_i, 2] == 0
            all_labels[-1][src_i, 2] = 1

    current_ids = segmented_input_ids
    early_exit = False

    norm_parts = [
        tokenizer.encode(part, add_special_tokens=False)
        for part in norm_word.split("-")
    ]
    norm_parts[-1].append(tokenizer.eos_token_id)

    while current_ids != norm_input_ids:
        if early_exit:
            all_labels.append(np.zeros((len(current_ids), 3), dtype=np.uint8))
            all_input_ids.append(current_ids)
            early_exit = False

        current_parts = []
        current_offsets = []
        part = []
        for i in current_ids:
            if i == DASH_GROUP[0]:
                current_offsets.append(
                    sum(len(x) for x in current_parts) + len(current_parts)
                )
                current_parts.append(part)
                part = []
                continue

            part.append(i)

        current_offsets.append(sum(len(x) for x in current_parts) + len(current_parts))
        current_parts.append(part)

        if len(all_labels) == 1:
            current_offsets = [o - i for i, o in enumerate(current_offsets)]

        for s_part, n_part, offset in zip(current_parts, norm_parts, current_offsets):
            for name, src_i, tgt_i in byte_edit_ops(s_part, n_part):
                src_i = offset + src_i

                if name == "insert":
                    if all_labels[-1][src_i, 0] != 0:
                        early_exit = True
                        continue

                    all_labels[-1][src_i, 0] = n_part[tgt_i]
                elif name == "replace":
                    if all_labels[-1][src_i, 1] != 0:
                        early_exit = True
                        continue
                    all_labels[-1][src_i, 1] = n_part[tgt_i]
                elif name == "delete":
                    if all_labels[-1][src_i, 2] != 0:
                        early_exit = True
                        continue
                    all_labels[-1][src_i, 2] = 1

        if len(all_labels) == 1:
            current_ids = apply_edits(input_ids, all_labels[-1])
        else:
            current_ids = apply_edits(current_ids, all_labels[-1])

    return all_input_ids, all_labels

SPLIT_REGEX = re.compile("((?:\p{S}|\p{P})+|\p{Z}+|\n|\t|\r)")
SEPARATOR_REGEX = re.compile("^(\p{Z}|\n|\t|\r)+$")


def segment(text, max_token_length=1000):
    text = unicodedata.normalize("NFKC", text)
    assert "\r" not in text

    splits = re.split(SPLIT_REGEX, text)

    tokens = []
    prefix = False

    for split in splits:
        if len(split) == 0:
            continue

        if SEPARATOR_REGEX.match(split):
            prefix = True
        elif len(split) > 0:
            if len(split) > max_token_length:
                return []

            # temporary, since sentencepiece normalizes it anyway
            tokens.append((" " if prefix else "", split))
            prefix = False

    return tokens