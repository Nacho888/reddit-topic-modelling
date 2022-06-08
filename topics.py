from collections import Counter
import logging
import logging_factory
import time
import glob
import gensim
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn
import gensim.corpora as corpora
import matplotlib.pyplot as plt
from gensim import models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import Phrases
from gensim.models.phrases import Phraser
# from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation, NMF


logger_err = logging_factory.get_module_logger("topics_err", logging.ERROR)
logger = logging_factory.get_module_logger("topics", logging.DEBUG)


def compute_lda_scikit():
    for file in glob.glob("./data/*.txt"):
        logger.debug(f"(scikit-learn) Processing file: '{file}'")

        # Load data from .txt file
        data = []
        with open(file, "r", encoding="utf-8") as f:
            data = [row.rstrip("\n") for row in f]

        ############
        # Unigrams #
        ############

        # Bag of words
        cv = CountVectorizer(ngram_range=(1,1), max_features=1000, max_df=0.80, min_df=15)
        corpus_bow = cv.fit_transform(data)

        # TF-IDF
        tf_trans = TfidfTransformer()
        corpus_tfidf = tf_trans.fit_transform(corpus_bow)

        ###########
        # Bigrams #
        ###########

        # Bag of words
        cv_bigrams = CountVectorizer(ngram_range=(2,2), max_features=1000, max_df=0.80, min_df=15)
        corpus_bigram = cv_bigrams.fit_transform(data)

        # TF-IDF
        tf_trans_bigrams = TfidfTransformer()
        corpus_tfidf_bigrams = tf_trans_bigrams.fit_transform(corpus_bigram)

        # Declaration of models
        word_models = [[corpus_bigram, "bigram_bow", cv_bigrams],
        [corpus_tfidf_bigrams, "bigram_tfidf", tf_trans_bigrams],
        [corpus_bow, "bow", cv],
        [corpus_tfidf, "tfidf", tf_trans]]

        for i in range(len(word_models)):
            # Get the name of the model
            model_name = word_models[i][1]

            # Build model (LDA/NMF)
            start_time = time.time()

            if "tfidf" in model_name:
                logger.debug(f"Computing NMF model for '{model_name}'")
                model = NMF(n_components=10, random_state=100, alpha=.1, l1_ratio=.5)
            else:
                logger.debug(f"Computing LDA model for '{model_name}'")
                model = LatentDirichletAllocation(n_components=10, learning_method="online", random_state=0)
            model.fit(word_models[i][0])
            logger.debug(f"Elapsed time building the model ({model_name}): {time.time() - start_time} seconds")

            # Simple representation
            def display_topics(model, feature_names, no_top_words):
                for topic_idx, topic in enumerate(model.components_):
                    print(f"Topic {topic_idx}: {' '.join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])}")
            display_topics(model, word_models[i][2].get_feature_names_out(), 10)

            # Compute Perplexity
            if "tfidf" not in model_name:
                logger.debug(f"Log Likelihood: {model.score(word_models[i][0])}")
                logger.debug(f"Perplexity: {model.perplexity(word_models[i][0])}")

            # # Compute Coherence Score
            # coherence_model_lda = CoherenceModel(model=model, corpus=word_models[i][0], dictionary=word_models[i][2], coherence="c_v")
            # coherence_lda = coherence_model_lda.get_coherence()
            # logger.debug(f"Coherence Score ({model_name}): {coherence_lda}")

            # Output to HTML
            vis = pyLDAvis.sklearn.prepare(model, word_models[i][0], word_models[i][2])
            filename = file.split('.')[1].replace("/", "_").replace("\\", "_")
            pyLDAvis.save_html(vis, f"./output/lda{filename}_{model_name}_sckit.html")
            logger.debug(f"HTML out generated ({model_name})\n")


def compute_gensim():
    coherences = []

    for file in glob.glob("./data/*.txt"):
        logger.debug(f"(gensim) Processing file: '{file}'")

        coherence_for_file = {}

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
        word_models = [[corpus_bigram, "bigram_bow", data_dict_bigrams],
        [corpus_tfidf_bigrams, "bigram_tfidf", data_dict_bigrams],
        [corpus_bow, "bow", data_dict],
        [corpus_tfidf, "tfidf", data_dict]]

        for i in range(len(word_models)):
            # Get the name of the model
            model_name = word_models[i][1]

            # Build LDA model
            start_time = time.time()

            if "tfidf" in model_name:
                logger.debug(f"Computing LSI model for '{model_name}'")
                model = gensim.models.lsimodel.LsiModel(corpus=word_models[i][0], id2word=word_models[i][2], num_topics=10)
            else:
                logger.debug(f"Computing LDA model for '{model_name}'")
                model = gensim.models.ldamodel.LdaModel(corpus=word_models[i][0],
                                                    id2word=word_models[i][2],
                                                    num_topics=10,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=5000,
                                                    passes=5,
                                                    alpha="auto",
                                                    per_word_topics=True)
            logger.debug(f"Elapsed time building the model ({model_name}): {time.time() - start_time} seconds")

            # Simple representation
            logger.debug(model.show_topics(num_topics=10, num_words=10, log=False, formatted=True))

            # Compute Perplexity
            if "tfidf" not in model_name:
                logger.debug(f"Perplexity ({model_name}): {model.log_perplexity(word_models[i][0])}")

            # Compute Coherence Score
            coherence_model = CoherenceModel(model=model, corpus=word_models[i][0], dictionary=word_models[i][2], coherence="u_mass")
            coherence = coherence_model.get_coherence()
            coherence_str = f"Coherence Score ({model_name}): {coherence}"
            if "tfidf" not in model_name:
                coherence_str += "\n"
            logger.debug(coherence_str)

            coherence_for_file[model_name] = coherence

            # Output to HTML
            if "tfidf" not in model_name:
                vis = pyLDAvis.gensim_models.prepare(model, word_models[i][0], word_models[i][2])
                filename = file.split('.')[1].replace("/", "_").replace("\\", "_")
                pyLDAvis.save_html(vis, f"./output/lda{filename}_{model_name}_gensim.html")
                logger.debug(f"HTML out generated ({model_name})\n")

        coherences.append(coherence_for_file)

    print_coherences(coherences)


def print_coherences(coherences):
    filenames = ["all_subreddits", "all_subreddits_no_control", "control"]
    for i, coherence_dict in enumerate(coherences):
        coherence_values = list(coherence_dict.values())
        model_names = list(coherence_dict.keys())
        plt.bar(range(len(model_names)), coherence_values, tick_label=model_names)
        plt.title(f"Input data: {filenames[i]}")
        plt.xlabel("Model name")
        plt.ylabel("Model coherence score (u_mass)")
        plt.savefig(f"./output/coherence_bar_{filenames[i]}.png")


def extract_topics(params):
    if "gensim" in params:
        compute_gensim()
    else:
        compute_lda_scikit()


compute_gensim()
# compute_lda_scikit()
