import logging
import logging_factory
import datetime
import cleantext
import re
import swifter
import glob
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pandas import DataFrame


logger_err = logging_factory.get_module_logger("preprocessing_err", logging.ERROR)
logger = logging_factory.get_module_logger("preprocessing", logging.DEBUG)


pst = PorterStemmer()

try:
    stopwords_set = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords")
    stopwords_set = set(stopwords.words("english"))

pattern = re.compile(r'\b(' + r'|'.join(stopwords_set) + r')\b\s*')


def pre_process(text: str, stem: bool = True):
    """
    Given a string cleans it up and returns it once processed. Applies:

    1) Lowercase
    2) Punctuation, email, phone and url removal
    3) Conversion of unicode symbols
    4) Normalize whitespaces
    5) Stopwords removal (using nltk's english list)
    6) Stem words using a semi-aggressive stemmer (the classical Porter Stemmer)

    :param text: str - the string to be processed
    :return: str - the processed string
    """

    # Lowercase, removers, convert unicode symbols and normalize whitespaces
    processed = cleantext.clean(text, lower=True, fix_unicode=True, no_punct=True, no_urls=True, no_emails=True, no_phone_numbers=True, normalize_whitespace=True, no_line_breaks=True)

    # Remove stopwords
    processed = pattern.sub('', processed)

    # Stem words
    if stem:
        try:
            stemmed_words = [pst.stem(word) for word in word_tokenize(processed)]
        except LookupError:
            import nltk
            nltk.download("punkt")
            stemmed_words = [pst.stem(word) for word in word_tokenize(processed)]

    return " ".join(stemmed_words) if stem else processed


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


def preprocess_inputs(subreddit_control: str, only_text: bool = True):
    full_df = None
    full_no_control_df = None
    control_df = None

    for file in glob.glob("./data/*.jsonl"):
        df = pd.DataFrame(pd.read_json(file, lines=True))

        logger.debug(f"Input: '{file}' with {df.shape[0]} posts")

        df = clean_data_frame(df, only_text)

        df["text"] = df["text"].swifter.apply(lambda x: pre_process(x, False))

        logger.debug(f"After preprocessing: {df.shape[0]} posts\n")

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

    full_df.to_csv("./data/all_subreddits.txt", header=None, index=None, sep="\n", mode="w")
    full_no_control_df.to_csv("./data/all_subreddits_no_control.txt", header=None, index=None, sep="\n", mode="w")
    control_df.to_csv("./data/control.txt", header=None, index=None, sep="\n", mode="w")
