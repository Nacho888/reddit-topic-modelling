import logging
import logging_factory
import os
import time
import glob
import gensim
import json
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn
import gensim.corpora as corpora
import matplotlib.pyplot as plt
from gensim import models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import Phrases
from gensim.models.phrases import Phraser

logger_err = logging_factory.get_module_logger("topics_err", logging.ERROR)
logger = logging_factory.get_module_logger("topics", logging.DEBUG)


def get_gensim_models(file: str):
    # Load data from .txt file
    data = []
    with open(file, "r", encoding="utf-8") as f:
        data = [row.rstrip("\n") for row in f]

    # All sentences in one list: one document per entry
    data = [doc.split(" ") for doc in data]

    ############
    # Unigrams #
    ############

    data_dict = corpora.Dictionary(data)
    data_dict.filter_extremes(no_below=15, no_above=0.80, keep_n=1000)

    # Bag of words
    corpus_bow = [data_dict.doc2bow(doc) for doc in data]

    # TF-IDF
    tfidf = models.TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]

    ###########
    # Bigrams #
    ###########

    bigram = Phrases(data)
    bigram_phraser = Phraser(bigram)
    bigram_token = []
    for doc in data:
        bigram_token.append(bigram_phraser[doc])

    data_dict_bigrams = gensim.corpora.Dictionary(bigram_token)
    data_dict_bigrams.filter_extremes(no_below=15, no_above=0.6, keep_n=1000)

    # Bag of words
    corpus_bigram = [data_dict_bigrams.doc2bow(doc) for doc in bigram_token]

    # TF-IDF
    tfidf_bigrams = models.TfidfModel(corpus_bigram)
    corpus_tfidf_bigrams = tfidf_bigrams[corpus_bigram]

    # Declaration of models
    word_models = [[corpus_bigram, "bi_bow", data_dict_bigrams],
    [corpus_tfidf_bigrams, "bi_tfidf", data_dict_bigrams],
    [corpus_bow, "bow", data_dict],
    [corpus_tfidf, "tfidf", data_dict]]

    return word_models


def compute_gensim(preprocessed_as):
    coherences = []

    for file in glob.glob(f"./data/{preprocessed_as}/*.txt"):
        logger.debug(f"(gensim) Processing file: '{file}'")

        coherence_for_file = {}

        word_models = get_gensim_models(file)

        for i in range(len(word_models)):
            # Get the name of the word model
            word_model_name = word_models[i][1]

            # Build topic model
            start_time = time.time()

            if "tfidf" in word_model_name: # LSI only with TFIDF
                logger.debug(f"Computing LSI model for '{word_model_name}'")
                model = gensim.models.lsimodel.LsiModel(corpus=word_models[i][0], id2word=word_models[i][2], num_topics=10)
                logger.debug(f"Elapsed time building the model ({word_model_name}): {time.time() - start_time} seconds")
                coherence_for_file = get_gensim_model_stats(model, word_models, f"{word_model_name + '_lsi'}", file, i, coherence_for_file, preprocessed_as)
            # else: # LDA and HDP with Bag of Words
            #     logger.debug(f"Computing LDA model for '{word_model_name}'")
            #     model = gensim.models.ldamodel.LdaModel(corpus=word_models[i][0],
            #                                         id2word=word_models[i][2],
            #                                         num_topics=10,
            #                                         random_state=100,
            #                                         update_every=1,
            #                                         chunksize=5000,
            #                                         passes=5,
            #                                         alpha="auto",
            #                                         per_word_topics=True)
            #     logger.debug(f"Elapsed time building the model ({word_model_name + '_lda'}): {time.time() - start_time} seconds")
            #     coherence_for_file = get_gensim_model_stats(model, word_models, f"{word_model_name + '_lda'}", file, i, coherence_for_file, preprocessed_as)

            #     logger.debug(f"Computing HDP model for '{word_model_name}'")
            #     model = gensim.models.hdpmodel.HdpModel(corpus=word_models[i][0], id2word=word_models[i][2])
            #     model = model.suggested_lda_model()
            #     logger.debug(f"Elapsed time building the model ({word_model_name + '_hdp'}): {time.time() - start_time} seconds")
            #     coherence_for_file = get_gensim_model_stats(model, word_models, f"{word_model_name + '_hdp'}", file, i, coherence_for_file, preprocessed_as)

        coherences.append(coherence_for_file)

    with open(f"./output/{preprocessed_as}/coherences.json", "w+") as f:
        json.dump(coherences, f)
    print_coherences(coherences, preprocessed_as, glob.glob(f"./data/{preprocessed_as}/*.txt"))


def get_gensim_model_stats(model, word_models, word_model_name, file, i, coherence_for_file, preprocessed_as):
    # Simple representation
    logger.debug(model.show_topics(num_topics=10, num_words=10, log=False, formatted=True))

    # Compute Perplexity
    if "lsi" not in word_model_name:
        logger.debug(f"Perplexity ({word_model_name}): {model.log_perplexity(word_models[i][0])}")

    # Compute Coherence Score
    coherence_model = CoherenceModel(model=model, corpus=word_models[i][0], dictionary=word_models[i][2], coherence="u_mass")
    coherence = coherence_model.get_coherence()
    coherence_str = f"Coherence Score ({word_model_name}): {coherence}"
    if "lsi" in word_model_name:
        coherence_str += "\n"
    logger.debug(coherence_str)

    coherence_for_file[word_model_name] = coherence

    # Output to HTML
    if "lsi" not in word_model_name:
        vis = pyLDAvis.gensim_models.prepare(model, word_models[i][0], word_models[i][2])
        filename = file.split("\\")[1].split(".")[0]
        try:
            os.remove(f"./output/{preprocessed_as}/{filename}_{word_model_name}_gensim.html")
        except FileNotFoundError:
            pass
        pyLDAvis.save_html(vis, f"./output/{preprocessed_as}/{filename}_{word_model_name}_gensim.html")
        logger.debug(f"HTML out generated ({word_model_name})\n")

    return coherence_for_file


def print_coherences(coherences, preprocessed_as, files):
    filenames = [file.split("\\")[1].split(".")[0] for file in files]
    for i, coherence_dict in enumerate(coherences):
        coherence_values = list(coherence_dict.values())
        model_names = list(coherence_dict.keys())
        try:
            os.remove(f"./output/{preprocessed_as}/coherence_bar{filenames[i]}.png")
        except FileNotFoundError:
            pass
        plt.bar(range(len(model_names)), coherence_values)
        plt.xticks(range(len(model_names)), model_names, fontsize=9)
        plt.title(f"Input data: {filenames[i]}")
        plt.xlabel("Model name")
        plt.ylabel("Model coherence score (u_mass)")
        plt.savefig(f"./output/{preprocessed_as}/coherence_bar_{filenames[i]}.png")
        plt.clf()


def extract_topics(preprocessed_as):
    compute_gensim(preprocessed_as)

compute_gensim("base")
# compute_gensim("base_stem")
# compute_gensim("spacy")s
# compute_gensim("spacy_stem")