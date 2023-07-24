use flate2::bufread::GzDecoder;
use lazy_static::lazy_static;
use rayon::prelude::*;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::collections::HashSet;
use std::io::prelude::*;
use std::path::Path;
use std::{collections::HashMap, time::Duration};

// adapted from HuggingFace's mc4.py
macro_rules! get_data_url {
    ($language:expr, $split_suffix:expr, $index:expr, $n_shards:expr) => {
        format!(
            "https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/multilingual/c4-{}{}.tfrecord-{:05}-of-{:05}.json.gz",
            $language,
            $split_suffix,
            $index,
            $n_shards
        )
    };
}

const MAX_SHARDS_PER_LANGUAGE: usize = 100;

const NON_DASH_MULTIPLIER: usize = 2;

const MAX_LENGTH: usize = 64;

const SHARD_SAVE_INTERVAL: usize = 4;

const LANGUAGES: &[&str] = &[
    "af", "am", "ar", "az", "be", "bg", "bn", "ca", "ceb", "co", "cs", "cy", "da", "de", "el",
    "en", "eo", "es", "et", "eu", "fa", "fi", "fil", "fr", "fy", "ga", "gd", "gl", "gu", "ha",
    "haw", "hi", "hmn", "ht", "hu", "hy", "id", "ig", "is", "it", "iw", "ja", "jv", "ka", "kk",
    "km", "kn", "ko", "ku", "ky", "la", "lb", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr",
    "ms", "mt", "my", "ne", "nl", "no", "ny", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk",
    "sl", "sm", "sn", "so", "sq", "sr", "st", "su", "sv", "sw", "ta", "te", "tg", "th", "tr", "uk",
    "und", "ur", "uz", "vi", "xh", "yi", "yo", "zh", "zu",
];
//const LANGUAGES: &[&str] = &[
//    "af", "bg", "bn", "ca", "da", "el", "et", "fi", "gl", "he", "hu", "lt", "lv", "pt", "sk", "sq",
//    "sv", "th", "tr", "uk",
//];

lazy_static! {
    static ref COMPOUND_SPLIT_CHARS: HashSet<char> = vec!['-', '‐'].into_iter().collect();
}
lazy_static! {
    static ref PUNCT_REGEX: regex::Regex = regex::Regex::new(r"^\p{P}+$").unwrap();
    static ref SPLIT_REGEX: regex::Regex = regex::Regex::new(
        COMPOUND_SPLIT_CHARS
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join("|")
            .as_str(),
    )
    .unwrap();
}

#[derive(Deserialize, Debug)]
struct Shards {
    train: usize,
}

#[derive(Deserialize, Debug)]
struct Sample {
    text: String,
}

fn check(word: &str) -> Option<(String, bool)> {
    if word.contains(['.', '@', '{', '}', '/', '[', ']']) {
        return None;
    }
    if word.contains("--") {
        return None;
    }

    if word.len() > MAX_LENGTH {
        return None;
    }

    let has_dash = word.chars().any(|c| COMPOUND_SPLIT_CHARS.contains(&c));

    let word = if has_dash {
        let mut prev_c = word.chars().next().unwrap();
        for c in word.chars().skip(1) {
            if (c.is_uppercase() && !COMPOUND_SPLIT_CHARS.contains(&prev_c))
                || (c.is_numeric() && !prev_c.is_numeric())
            {
                return None;
            }
            prev_c = c;
        }

        let word = word.trim_matches(|x: char| PUNCT_REGEX.is_match(x.to_string().as_str()));

        for part in SPLIT_REGEX.split(word) {
            if part.chars().all(|c| c.is_uppercase() || c.is_numeric()) {
                return None;
            }
        }

        word
    } else {
        word.trim_matches(|x: char| PUNCT_REGEX.is_match(x.to_string().as_str()))
    };

    Some((word.to_string(), has_dash))
}

fn count(
    text: &str,
    dash_counter: &mut HashMap<String, usize>,
    non_dash_counter: &mut HashMap<String, usize>,
) {
    for word in text.split_whitespace() {
        if let Some((word, has_dash)) = check(word) {
            if has_dash {
                *dash_counter.entry(word.to_string()).or_insert(0) += 1;
            } else if non_dash_counter.len() < dash_counter.len() * NON_DASH_MULTIPLIER {
                *non_dash_counter.entry(word.to_string()).or_insert(0) += 1;
            }
        }
    }
}

fn save_counter(counter: &HashMap<String, usize>, path: &str) {
    let out_word_counter = counter
        .iter()
        .filter(|(_, v)| **v > 1)
        .collect::<HashMap<_, _>>();
    let mut file = std::fs::File::create(path).unwrap();
    serde_json::to_writer(&mut file, &out_word_counter).unwrap();
}

fn main() {
    let n_shards_per_split: HashMap<String, Shards> =
        serde_json::from_str(include_str!("N_SHARDS_PER_SPLIT.json")).unwrap();
    let n_train_shards = n_shards_per_split
        .into_iter()
        .map(|(k, v)| (k, v.train))
        .collect::<HashMap<_, _>>();

    LANGUAGES.par_iter().for_each(|&language| {
        let n_shards = n_train_shards.get(language).unwrap();
        let n_shards_to_obtain = std::cmp::min(*n_shards, MAX_SHARDS_PER_LANGUAGE);

        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .unwrap();

        let mut dash_counter = HashMap::new();
        let mut non_dash_counter = HashMap::new();

        if Path::new(format!("data/{}_final_dash.json", language).as_str()).exists() {
            return;
        }

        let mut start = 0;
        for i in 0..n_shards_to_obtain {
            if Path::new(format!("data/{}_{}_dash.json", language, i).as_str()).exists() {
                start = i;
                dash_counter = serde_json::from_str(
                    &std::fs::read_to_string(format!("data/{}_{}_dash.json", language, i).as_str())
                        .unwrap(),
                )
                .unwrap();
                non_dash_counter = serde_json::from_str(
                    &std::fs::read_to_string(
                        format!("data/{}_{}_non_dash.json", language, i).as_str(),
                    )
                    .unwrap(),
                )
                .unwrap();
            }
        }

        for i in start..n_shards_to_obtain {
            println!(
                "Processing shard {} of {} for {}...",
                i, n_shards_to_obtain, language
            );

            let response = client
                .get(get_data_url!(language, "", i, n_shards))
                .send()
                .unwrap();
            let bytes = response.bytes().unwrap();

            println!(
                "Downloaded shard {} of {} for {}...",
                i, n_shards_to_obtain, language
            );

            let mut gz = GzDecoder::new(bytes.as_ref());
            let mut s = String::new();
            gz.read_to_string(&mut s).unwrap();

            for line in s.lines() {
                let sample: Sample = serde_json::from_str(line).unwrap();
                count(&sample.text, &mut dash_counter, &mut non_dash_counter);
            }
            println!(
                "Finished shard {} of {} for {}...",
                i, n_shards_to_obtain, language
            );

            if (i + 1) % SHARD_SAVE_INTERVAL == 0 {
                save_counter(
                    &dash_counter,
                    format!("data/{}_{}_dash.json", language, i).as_str(),
                );
                save_counter(
                    &non_dash_counter,
                    format!("data/{}_{}_non_dash.json", language, i).as_str(),
                );
            }
        }

        save_counter(
            &dash_counter,
            format!("data/{}_final_dash.json", language).as_str(),
        );
        save_counter(
            &non_dash_counter,
            format!("data/{}_final_non_dash.json", language).as_str(),
        );
    });
}
