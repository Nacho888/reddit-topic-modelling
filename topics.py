import logging
import logging_factory
import time
import glob
import gensim
import pyLDAvis
import pyLDAvis.gensim_models
import gensim.corpora as corpora
from gensim import models
from gensim.models.coherencemodel import CoherenceModel


logger_err = logging_factory.get_module_logger("topics_err", logging.ERROR)
logger = logging_factory.get_module_logger("topics", logging.DEBUG)


def compute_lda():
    for file in glob.glob("./data/*.txt"):
        logger.debug(f"Processing file: '{file}'")
        data = []
        with open(file, "r", encoding="utf-8") as f:
            data = [row.rstrip("\n") for row in f]
        data = [doc.split(" ") for doc in data]

        data_dict = corpora.Dictionary(data)
        data_dict.filter_extremes(no_below=15, no_above=0.75, keep_n=1000)
        # Bag of words
        corpus_bow = [data_dict.doc2bow(doc) for doc in data]
        # TF-IDF
        tfidf = models.TfidfModel(corpus_bow)
        corpus_tfidf = tfidf[corpus_bow]
        # Bigrams
        bigram = gensim.models.Phrases(data, min_count=15, threshold=10)
        bigram_mod = gensim.models.phrases.Phraser(bigram)

        word_models = [corpus_bow, corpus_tfidf, bigram_mod]
        word_models_names = ["bow", "tfidf", "bigram"]

        for i in range(len(word_models)):
            model_name = word_models_names[i]
            logger.debug(f"Computing LDA model for '{model_name}'")
            start_time = time.time()
            # Build LDA model
            lda_model = gensim.models.ldamodel.LdaModel(corpus=word_models[i],
                                                    id2word=data_dict,
                                                    num_topics=10,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=5000,
                                                    passes=5,
                                                    alpha="auto",
                                                    per_word_topics=True)
            logger.debug(f"Elapsed time building the model ({model_name}): {time.time() - start_time} seconds")
            lda_model.print_topics()

            # Compute Perplexity
            logger.debug(f"Perplexity ({model_name}): {lda_model.log_perplexity(word_models[i])}")

            # Compute Coherence Score
            coherence_model_lda = CoherenceModel(model=lda_model, corpus=word_models[i], dictionary=data_dict, coherence="u_mass")
            coherence_lda = coherence_model_lda.get_coherence()
            logger.debug(f"Coherence Score ({model_name}): {coherence_lda}")

            vis = pyLDAvis.gensim_models.prepare(lda_model, word_models[i], data_dict)
            filename = file.split('.')[1].replace("/", "_").replace("\\", "_")
            pyLDAvis.save_html(vis, f"./output/lda{filename}_{model_name}.html")
            logger.debug(f"HTML out generated ({model_name})\n")


def extract_topics():
    compute_lda()

extract_topics()