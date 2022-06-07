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
             logger_err.error("Arguments expected as: -pre <subreddit_control>")
        else:
            preprocess_inputs()
        preprocess_inputs(subreddit_control)
    elif option == "-top":
        extract_topics()
    else:
        logger_err.error("No options provided in the arguments: \n\t* '-pre' (to generate .txt files given submission .jsonl files) \n\t* '-top' (to generate the .html files with all the topics given the .txt files)")


if __name__ == "__main__":
        main(sys.argv[1:])
