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
