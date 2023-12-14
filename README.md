# CompoundPiece

Code, models and dataset for the paper [CompoundPiece: Evaluating and Improving Decompounding Performance of Language Models](https://arxiv.org/abs/2305.14214).

## Models

- https://huggingface.co/benjamin/compoundpiece is the Stage1+Stage2 trained ByT5 model.
- https://huggingface.co/benjamin/compoundpiece-stage1 is the model with (self-supervised) Stage1 training only.

## Dataset

- https://huggingface.co/datasets/benjamin/compoundpiece contains the dataset of web-scraped words and the Wiktionary dataset.

## Tokenizers

`tokenizers/` contains the tokenizers used in the paper.
  - `tokenizers/baseline` are the regular SPM tokenizers.
  - `tokenizers/compoundpiece` the CompoundPiece tokenizers.

## Code

- `get_web_full_text_data.py`, `get_web_word_data.py` and `get_wiktionary_data.py` are the scripts to obtain their respective datasets.
  - `get_web_word_data.py` ingests JSON files which must previously be prepared via `mc4_words/src/main.rs`.
- `word_segmentation/train_spm.py` can be used to train SPM models with or without CompoundPiece pretokenization.
  - This internally runs inference via [`t5x`](https://github.com/google-research/t5x) so you need to clone the t5x repo.
  - It also needs model checkpoints in t5x models, find these in the `t5x/` directory of the HF model repositories, for example here: https://huggingface.co/benjamin/compoundpiece/tree/main/t5x.
  - The text corpus to train on is expected to be in tfrecord format. Prepare it with this command to reproduce the results from the paper: `python get_web_full_text_data.py --n_shards=1 --n_train_pages=10000000 --out_train_dir=<train_dir> --out_valid_file=<valid_file>`.
- `word_segmentation/t5x` contains the configs and scripts to train the models from the paper with t5x.

## Citation

Please cite CompoundPiece as 

```
@inproceedings{minixhofer-etal-2023-compoundpiece,
    title = "{C}ompound{P}iece: Evaluating and Improving Decompounding Performance of Language Models",
    author = "Minixhofer, Benjamin  and
      Pfeiffer, Jonas  and
      Vuli{\'c}, Ivan",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.24",
    pages = "343--359",
    abstract = "While many languages possess processes of joining two or more words to create compound words, previous studies have been typically limited only to languages with excessively productive compound formation (e.g., German, Dutch) and there is no public dataset containing compound and non-compound words across a large number of languages. In this work, we systematically study decompounding, the task of splitting compound words into their constituents, at a wide scale. We first address the data gap by introducing a dataset of 255k compound and non-compound words across 56 diverse languages obtained from Wiktionary. We then use this dataset to evaluate an array of Large Language Models (LLMs) on the decompounding task. We find that LLMs perform poorly, especially on words which are tokenized unfavorably by subword tokenization. We thus introduce a novel methodology to train dedicated models for decompounding. The proposed two-stage procedure relies on a fully self-supervised objective in the first stage, while the second, supervised learning stage optionally fine-tunes the model on the annotated Wiktionary data. Our self-supervised models outperform the prior best unsupervised decompounding models by 13.9{\%} accuracy on average. Our fine-tuned models outperform all prior (language-specific) decompounding tools. Furthermore, we use our models to leverage decompounding during the creation of a subword tokenizer, which we refer to as CompoundPiece. CompoundPiece tokenizes compound words more favorably on average, leading to improved performance on decompounding over an otherwise equivalent model using SentencePiece tokenization.",
}
```

## Acknowledgments

Ivan Vulić is supported by a personal Royal Society University Research Fellowship ‘Inclusive and Sustainable Language Technology for a Truly Multilingual World’ (no 221137; 2022–). Research supported with Cloud TPUs from Google’s TPU Research Cloud (TRC). We thank Sebastian Ruder and Srini Narayanan for helpful feedback on a draft of the paper.
