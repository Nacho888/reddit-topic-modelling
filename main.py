import logging
import logging_factory
import sys
from preprocessing import preprocess_inputs


logger_err = logging_factory.get_module_logger("main_err", logging.ERROR)
logger = logging_factory.get_module_logger("main", logging.DEBUG)


def main(argv):
    subreddit_control = argv[0]
    preprocess_inputs(subreddit_control)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        logger_err.error("No subreddit control provided in the arguments")