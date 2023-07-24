import seqio
import t5.data
import tensorflow as tf
from functools import partial
import numpy as np
import os
from word_segmentation.utils import LANGUAGES


def get_output_features(vocab):
    return {
        "inputs": seqio.Feature(vocabulary=vocab, add_eos=True, required=False),
        "targets": seqio.Feature(vocabulary=vocab, add_eos=True),
    }


MT5_REPO_DIR = "<your path>" # path to https://github.com/google-research/multilingual-t5

import sys

sys.path.append(MT5_REPO_DIR)

import multilingual_t5
from multilingual_t5.evaluation import metrics as mt5_metrics

MT5_VOCAB = seqio.SentencePieceVocabulary(
    "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"
)
OUR_MT5_VOCAB = seqio.SentencePieceVocabulary(
    "tokenizers/compoundpiece/multilingual.model"
)
BASELINE_MT5_VOCAB = seqio.SentencePieceVocabulary(
    "tokenizers/baseline/multilingual.model"
)


def accuracy(targets, predictions):
    return {"accuracy": np.mean([x == y for x, y in zip(targets, predictions)])}


seqio.TaskRegistry.add(
    "word_segmentation_fewshot",
    source=seqio.TFExampleDataSource(
        {"test": "data/data.tfrecord"},
        {
            "inputs": tf.io.FixedLenFeature([], dtype=tf.string),
            "targets": tf.io.FixedLenFeature([], dtype=tf.string),
        },
    ),
    preprocessors=[
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[],
    output_features=get_output_features(t5.data.get_default_vocabulary()),
)

seqio.TaskRegistry.add(
    "word_segmentation_byte_pretrain",
    source=seqio.TextLineDataSource(
        {"train": "data/web/train.tsv", "validation": "data/web/valid.tsv"},
    ),
    preprocessors=[
        partial(t5.data.preprocessors.preprocess_tsv, num_fields=3),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[accuracy],
    output_features=get_output_features(t5.data.ByteVocabulary()),
)

seqio.TaskRegistry.add(
    "word_segmentation_byte_finetune",
    source=seqio.TextLineDataSource(
        {"train": "data/wiktionary/train.tsv", "validation": "data/wiktionary/valid.tsv"},
    ),
    preprocessors=[
        partial(t5.data.preprocessors.preprocess_tsv, num_fields=5),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[accuracy],
    output_features=get_output_features(t5.data.ByteVocabulary()),
)

seqio.TaskRegistry.add(
    "word_segmentation_byte_infer",
    source=seqio.TextLineDataSource(
        {"test": os.environ.get("T5X_TEST_TSV")},
    ),
    preprocessors=[
        partial(
            t5.data.preprocessors.preprocess_tsv, num_fields=1, targets_format="{0}"
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=get_output_features(t5.data.ByteVocabulary()),
)

seqio.TaskRegistry.add(
    "word_segmentation_t5_pretrain",
    source=seqio.TextLineDataSource(
        {"train": "data/web/train.tsv", "validation": "data/web/valid.tsv"},
    ),
    preprocessors=[
        partial(t5.data.preprocessors.preprocess_tsv, num_fields=3),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[accuracy],
    output_features=get_output_features(t5.data.get_default_vocabulary()),
)

seqio.TaskRegistry.add(
    "word_segmentation_t5_finetune",
    source=seqio.TextLineDataSource(
        {"train": "data/wiktionary/train.tsv", "validation": "data/wiktionary/valid.tsv"},
    ),
    preprocessors=[
        partial(t5.data.preprocessors.preprocess_tsv, num_fields=5),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[accuracy],
    output_features=get_output_features(t5.data.get_default_vocabulary()),
)

seqio.TaskRegistry.add(
    "word_segmentation_mt5_pretrain",
    source=seqio.TextLineDataSource(
        {"train": "data/web/train.tsv", "validation": "data/web/valid.tsv"},
    ),
    preprocessors=[
        partial(t5.data.preprocessors.preprocess_tsv, num_fields=3),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[accuracy],
    output_features=get_output_features(MT5_VOCAB),
)

seqio.TaskRegistry.add(
    "word_segmentation_mt5_finetune",
    source=seqio.TextLineDataSource(
        {"train": "data/wiktionary/train.tsv", "validation": "data/wiktionary/valid.tsv"},
    ),
    preprocessors=[
        partial(t5.data.preprocessors.preprocess_tsv, num_fields=5),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[accuracy],
    output_features=get_output_features(MT5_VOCAB),
)

seqio.TaskRegistry.add(
    "our_c4_v220_span_corruption",
    source=seqio.TFExampleDataSource(
        {
            "train": [
                f"/mnt/disks/persist/data/raw_text/train/shard{i}.tfrecord"
                for i in range(100)
            ],
            "valid": "/mnt/disks/persist/data/raw_text/valid.tfrecord",
        },
        {
            "lang": tf.io.FixedLenFeature([], dtype=tf.string),
            "text": tf.io.FixedLenFeature([], dtype=tf.string),
        },
        reader_cls=partial(
            tf.data.TFRecordDataset,
            compression_type="GZIP",
            num_parallel_reads=tf.data.AUTOTUNE,
        ),
    ),
    preprocessors=[
        partial(seqio.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=get_output_features(OUR_MT5_VOCAB),
    metric_fns=[],
)

seqio.TaskRegistry.add(
    "baseline_c4_v220_span_corruption",
    source=seqio.TFExampleDataSource(
        {
            "train": [
                f"/mnt/disks/persist/data/raw_text/train/shard{i}.tfrecord"
                for i in range(100)
            ],
            "valid": "/mnt/disks/persist/data/raw_text/valid.tfrecord",
        },
        {
            "lang": tf.io.FixedLenFeature([], dtype=tf.string),
            "text": tf.io.FixedLenFeature([], dtype=tf.string),
        },
        reader_cls=partial(
            tf.data.TFRecordDataset,
            compression_type="GZIP",
            num_parallel_reads=tf.data.AUTOTUNE,
        ),
    ),
    preprocessors=[
        partial(seqio.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=get_output_features(BASELINE_MT5_VOCAB),
    metric_fns=[],
)

## extrinsic eval

seqio.TaskRegistry.add(
    "word_segmentation_mt5_our_finetune",
    source=seqio.TextLineDataSource(
        {"train": "data/wiktionary/train.tsv", "validation": "data/wiktionary/valid.tsv"},
    ),
    preprocessors=[
        partial(t5.data.preprocessors.preprocess_tsv, num_fields=5),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[accuracy],
    output_features=get_output_features(OUR_MT5_VOCAB),
)

seqio.TaskRegistry.add(
    "word_segmentation_mt5_baseline_finetune",
    source=seqio.TextLineDataSource(
        {"train": "data/wiktionary/train.tsv", "validation": "data/wiktionary/valid.tsv"},
    ),
    preprocessors=[
        partial(t5.data.preprocessors.preprocess_tsv, num_fields=5),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[accuracy],
    output_features=get_output_features(BASELINE_MT5_VOCAB),
)

import sys

sys.path.append("~/multilingual-")


# from mt5
def create_wikiann_ner_tasks_and_mixtures(task_prefix, task_suffix, output_features):
    NER_LANGS = [
        "af",
        "bg",
        "bn",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fr",
        "he",
        "hi",
        "hu",
        "id",
        "it",
        "ka",
        "kk",
        "ml",
        "nl",
        "pt",
        "ru",
        "ta",
        "te",
        "th",
        "tr",
        "yo",
    ]

    for lang in NER_LANGS:
        seqio.TaskRegistry.add(
            f"{task_prefix}ner_train{task_suffix}.{lang}",
            source=seqio.TfdsDataSource(
                tfds_name="wikiann/{}:1.0.0".format(lang), splits=["train"]
            ),
            preprocessors=[
                multilingual_t5.preprocessors.wikiann,
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim,
            ],
            output_features=output_features,
            metric_fns=[mt5_metrics.span_f1],
        )

        seqio.TaskRegistry.add(
            f"{task_prefix}ner_eval{task_suffix}.{lang}",
            source=seqio.TfdsDataSource(
                tfds_name="wikiann/{}:1.0.0".format(lang), splits=["validation", "test"]
            ),
            preprocessors=[
                multilingual_t5.preprocessors.wikiann,
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim,
            ],
            output_features=output_features,
            metric_fns=[mt5_metrics.span_f1],
        )

    # NER zero-shot
    seqio.MixtureRegistry.add(
        f"{task_prefix}ner_zeroshot{task_suffix}",
        [f"{task_prefix}ner_train{task_suffix}.en"],
        default_rate=1.0,
    )

    # NER multilingual
    seqio.MixtureRegistry.add(
        f"{task_prefix}ner_multilingual{task_suffix}",
        [f"{task_prefix}ner_train{task_suffix}.{lang}" for lang in NER_LANGS],
        default_rate=1.0,
    )

create_wikiann_ner_tasks_and_mixtures(
    task_prefix="our_mt5_",
    task_suffix="",
    output_features=get_output_features(OUR_MT5_VOCAB),
)

create_wikiann_ner_tasks_and_mixtures(
    task_prefix="baseline_mt5_",
    task_suffix="",
    output_features=get_output_features(BASELINE_MT5_VOCAB),
)