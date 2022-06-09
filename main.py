import logging
import logging_factory
import sys
from preprocessing import preprocess_inputs
from topics import extract_topics


logger_err = logging_factory.get_module_logger("main_err", logging.ERROR)
logger = logging_factory.get_module_logger("main", logging.DEBUG)


def main(argv):
    option = argv[0]
    if option == "-pre":
        subreddit_control = argv[1]
        if subreddit_control is None or subreddit_control == "":
             logger_err.error("Arguments expected as:\n\t-pre <subreddit_control: string> <use spacy: True/False> <stem: True/False>")
        else:
            args = argv[2:]
            if len(args) < 2 or len(args) > 2:
                logger_err.error("Arguments expected as:\n\t-pre <subreddit_control: string> <use spacy: True/False> <stem: True/False>")
            try:
                preprocess_inputs(subreddit_control, bool(args[0]), bool(args[1]))
            except Exception:
                logger_err.error("Arguments expected as:\n\t-pre <subreddit_control: string> <use spacy: True/False> <stem: True/False>")
        preprocess_inputs(subreddit_control)
    elif option == "-top":
        preprocessed_as = argv[1]
        if preprocessed_as is None or preprocessed_as == "":
            logger_err.error("Arguments expected as:\n\t-top <type of preprocessing: string (base | base_stem | spacy | spacy_stem)>")
        else:
            try:
                extract_topics(preprocessed_as)
            except Exception:
                logger_err.error("Arguments expected as:\n\t-top <type of preprocessing: string (base | base_stem | spacy | spacy_stem)>")
    else:
        logger_err.error("No options provided in the arguments:\n\t* '-pre' (to generate .txt files given submission .jsonl files) \n\t* '-top' (to generate the .html files with all the topics given the .txt files)")


if __name__ == "__main__":
    main(sys.argv[1:])

    # preprocess_inputs("ImmigrationCanada", True, False, False)
    # preprocess_inputs("ImmigrationCanada", True, False, True)
    # preprocess_inputs("ImmigrationCanada", True, True, False)
    # preprocess_inputs("ImmigrationCanada", True, True, True)

    # extract_topics("base")
    # extract_topics("base_stem")
    # extract_topics("spacy")
    # extract_topics("spacy_stem")
