import logging
import logging_factory
import time
import glob
import gensim
import pyLDAvis
import pyLDAvis.gensim_models
import gensim.corpora as corpora
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
        doc_term_matrix = [data_dict.doc2bow(doc) for doc in data]

        start_time = time.time()
        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=doc_term_matrix,
                                                id2word=data_dict,
                                                num_topics=10,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=5000,
                                                passes=5,
                                                alpha="auto",
                                                per_word_topics=True)
        print(f"Elapsed time building the model: {time.time() - start_time} seconds")
        lda_model.print_topics()

        # Compute Perplexity
        print(f"Perplexity: {lda_model.log_perplexity(doc_term_matrix)}")

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, corpus=doc_term_matrix, dictionary=data_dict, coherence="u_mass")
        coherence_lda = coherence_model_lda.get_coherence()
        print(f"Coherence Score: {coherence_lda}\n")

        vis = pyLDAvis.gensim_models.prepare(lda_model, doc_term_matrix, data_dict)
        filename = file.split('.')[1].replace("/", "_").replace("\\", "_")
        pyLDAvis.save_html(vis, f"./output/lda{filename}.html")


def extract_topics():
    compute_lda()

extract_topics()