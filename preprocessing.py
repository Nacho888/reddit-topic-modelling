from concurrent.futures import process
import logging
import logging_factory
import datetime
import cleantext
import re
import swifter
import glob
import numpy as np
import pandas as pd
import spacy
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pandas import DataFrame


logger_err = logging_factory.get_module_logger("preprocessing_err", logging.ERROR)
logger = logging_factory.get_module_logger("preprocessing", logging.DEBUG)

try:
    stopwords_set = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords")
    stopwords_set = set(stopwords.words("english"))

pattern = re.compile(r'\b(' + r'|'.join(stopwords_set) + r')\b\s*')
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
pst = PorterStemmer()


def pre_process(text: str, spacy: bool = True, stem: bool = False):
    """
    Given a string cleans it up and returns it once processed. Applies:

    1) Lowercase
    2) Several removers (punctuation, numbers, symbols, emails...)
    3) Conversion of unicode symbols
    4) Normalize whitespaces
    5) Stopwords removal (using nltk's english list)
    6) Stem words using a semi-aggressive stemmer (the classical Porter Stemmer) - optional

    :param text: str - the string to be processed
    :param stem: bool - whether to stem the words or not
    :return: str - the processed string
    """

    # Lowercase, removers, convert unicode symbols and normalize whitespaces
    processed = cleantext.clean(text, lower=True, fix_unicode=True, no_punct=True, no_urls=True, no_emails=True, no_phone_numbers=True, normalize_whitespace=True, no_line_breaks=True, no_digits=True, no_numbers=True, no_emoji=True, replace_with_number="", replace_with_digit="")

    # Remove stopwords
    processed = pattern.sub('', processed)

    if spacy:
        doc = nlp(processed)
        processed = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB", "ADV"]]
    else:
        processed = processed.split(" ")

    # Stem words
    if stem:
        try:
            stemmed_words = [pst.stem(word) for word in processed]
        except LookupError:
            import nltk
            nltk.download("punkt")
            stemmed_words = [pst.stem(word) for word in processed]

    return " ".join(stemmed_words) if stem else " ".join(processed)


def clean_data_frame(df: DataFrame, remove_all_fields: bool = False):
    # Substitute reddit's keyword '[removed], [deleted] or nan' with empty string
    df["title"] = np.where((df.title == "[removed]") |
                                (df.title == "[deleted]") |
                                (df.title == np.nan),'', df.title)
    df["selftext"] = np.where((df.selftext == "[removed]") |
                                    (df.selftext == "[deleted]") |
                                    (df.selftext == np.nan),'', df.selftext)

    # Remove rows with no text in title neither selftext
    df = df[df[["title", "selftext"]].ne('').all(axis=1)]

    # Join the rest
    df_copy = df.copy()
    df_copy["text"] = df["title"] + " " + df["selftext"]

    # Leave only the text field
    if remove_all_fields:
        df_copy = df_copy[["text"]]

    return df_copy


def preprocess_inputs(subreddit_control: str, only_text: bool = True, spacy: bool = True, stem: bool = False):
    full_df = None
    full_no_control_df = None
    control_df = None

    for file in glob.glob("./data/*.jsonl"):
        df = pd.DataFrame(pd.read_json(file, lines=True))

        logger.debug(f"Input: '{file}' with {df.shape[0]} posts")

        df = clean_data_frame(df, only_text)
        logger.debug(f"After post cleanup: {df.shape[0]} posts")

        df["text"] = df["text"].swifter.apply(lambda x: pre_process(x, spacy=spacy, stem=stem))

        if subreddit_control in file:
            control_df = df

        if full_df is None:
            full_df = df
        else:
            full_df = full_df.append(df, ignore_index=True)

        if full_no_control_df is None and subreddit_control not in file:
            full_no_control_df = df
        elif subreddit_control not in file:
            full_no_control_df = full_no_control_df.append(df, ignore_index=True)

    logger.debug(f"SIZES - All posts: {full_df.shape[0]}, control ({subreddit_control}): {control_df.shape[0]}, all posts no control: {full_no_control_df.shape[0]}")

    path = "spacy" if spacy else "base"
    path += "_stem/" if stem else "/"

    full_df.to_csv(f"./data/{path}all_subreddits.txt", header=None, index=None, sep="\n", mode="w")
    full_no_control_df.to_csv(f"./data/{path}all_subreddits_no_control.txt", header=None, index=None, sep="\n", mode="w")
    control_df.to_csv(f"./data/{path}control.txt", header=None, index=None, sep="\n", mode="w")


preprocess_inputs("ImmigrationCanada", True, True, True)
