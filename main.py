import argparse
import importlib
import logging.config
import os
import sys
import time
from typing import Callable

import yaml
from chatterbot import ChatBot
from chatterbot.comparisons import (
    JaccardSimilarity,
    LevenshteinDistance,
    SpacySimilarity,
)
from chatterbot.trainers import ChatterBotCorpusTrainer

from data import DataWrapper, OpenSubtitles, PersonaChat, WizardOfWikipedia
from utils import (
    ConditionalFormatter,
    LoggerWriter,
    Split,
    convert_chatterbot_filename,
)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("chitchat")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = ConditionalFormatter("[%(asctime)-15s] %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logging.StreamHandler.terminator = ""
    sys.stdout = LoggerWriter(logger.info)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage="train_and_prompt.py [-h] (-p | -w | -s -l LANG) [-f PERCENTAGE] (--spacy | --jaccard | --levenshtein | --flair) [-t [-e]] [-n NAME]",
        description="Utility for training a ChatBot and prompting it.",
    )
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "-p",
        "--persona-chat",
        action="store_true",
        help="Train the ChatBot with PersonaChat dataset.",
    )
    dataset_group.add_argument(
        "-w",
        "--wizard-of-wikipedia",
        action="store_true",
        help="Train the ChatBot with WizardOfWikipedia dataset.",
    )
    dataset_group.add_argument(
        "-s",
        "--open-subtitles",
        action="store_true",
        help="Train the ChatBot with OpenSubtitles dataset.",
    )

    parser.add_argument(
        "-f",
        "--dataset-fraction",
        type=float,
        help="Select the fraction of the dataset used to train and test the ChatBot. The value must be a real number in the range (0,1].",
        default=0.1,
    )

    comparison_group = parser.add_mutually_exclusive_group(required=True)
    comparison_group.add_argument(
        "--spacy",
        action="store_true",
        help="Use SpaCy similarity (word2vec averaging).",
    )
    comparison_group.add_argument(
        "--jaccard", action="store_true", help="Use Jaccard similarity."
    )
    comparison_group.add_argument(
        "--levenshtein", action="store_true", help="Use Levenshtein distance."
    )
    comparison_group.add_argument(
        "--flair", action="store_true", help="Use Flair document embeddings."
    )

    parser.add_argument(
        "-t",
        "--run-test",
        action="store_true",
        help="Run ChatBot against dataset test split.",
    )
    parser.add_argument(
        "-n", "--name", type=str, help="ChatBot name.", default="chitchat-bot"
    )

    args, _ = parser.parse_known_args()
    if args.open_subtitles:
        parser.add_argument(
            "-l",
            "--language",
            type=str,
            help="Select the language of the OpenSubtitles dataset (ISO 639-1 language code).",
            required=True,
        )
    if args.run_test:
        parser.add_argument(
            "-e", "--evaluate", action="store_true", help="Evaluate trained ChatBot."
        )

    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> DataWrapper:
    if args.persona_chat:
        dataset = PersonaChat()
    elif args.wizard_of_wikipedia:
        dataset = WizardOfWikipedia()
    elif args.open_subtitles:
        dataset = OpenSubtitles(lang=args.language)
    else:
        raise ValueError("Please, specify a dataset to train the ChatBot.")

    return dataset


def get_comparison_function(args: argparse.Namespace) -> Callable:
    if args.spacy:
        comparison_function = SpacySimilarity
    elif args.jaccard:
        comparison_function = JaccardSimilarity
    elif args.levenshtein:
        comparison_function = LevenshteinDistance
    elif args.flair:
        raise NotImplementedError
    else:
        raise ValueError(
            "Please, specify a comparison function to select among ChatBot's response candidates."
        )

    return comparison_function


def log_line() -> None:
    logger.info("\n")
    logger.info("-----------------------------------------------------\n")
    logger.info("\n")


def enter_interactive_mode(bot: ChatBot) -> None:
    log_line()
    logger.info(
        "Entering interactive mode. Press CTRL+C to start a new conversation, and CTRL+D to exit.\n"
    )
    conversation_id = 0
    sys.stdout = (
        sys.__stdout__
    )  # Need to reset sys.stdout so that input() does not create an extra newline
    while True:
        try:
            logger.info("(YOU) >>> ")
            txt = input()
            if txt.strip() == "":
                logger.warning("Empty input. Please, enter some text...\n")
                continue
            response_stmt = bot.get_response(
                txt, conversation=f"user:conv-{conversation_id}"
            )
            logger.info(f"(BOT) >>> {response_stmt.text}\n")
        except KeyboardInterrupt:
            log_line()
            logger.info("Starting a new conversation.\n")
            log_line()
            conversation_id += 1
        except (EOFError, SystemExit):
            log_line()
            logger.info("Exiting interactive mode.\n")
            break


def run_against_test(bot: ChatBot, dataset: DataWrapper) -> None:
    test_filenames = dataset.get_dataset_split(Split.test)
    log_line()
    logger.info(
        f"Running ChatBot against test partition ({len(test_filenames)} files)\n"
    )
    with open(f"test-responses-{dataset.dataset_percentage}.txt", "w") as outfp:
        for filename in test_filenames:
            full_filename = convert_chatterbot_filename(filename)
            infp = open(full_filename, "r")
            contents = yaml.safe_load(infp)
            for idx, conversation in enumerate(contents["conversations"]):
                for utterance in conversation[:-1]:
                    response_stmt = bot.get_response(
                        utterance, conversation=f"test:conv-{idx}"
                    )
                    outfp.write(f"{response_stmt.text}\n")

                percent = float(idx + 1) / len(contents["conversations"])
                hashes = "#" * int(round(percent * 20))
                spaces = " " * (20 - len(hashes))
                logger.info(
                    f"Testing {os.path.basename(full_filename)}: [{hashes + spaces}] {int(round(percent * 100))}%\r"
                )
            logger.info(
                f"Testing {os.path.basename(full_filename)}: [{'#' * 20}] 100%\n"
            )
            infp.close()


def run_evaluation(bot: ChatBot, dataset: DataWrapper, **kwargs) -> None:
    log_line()
    logger.info("Evaluating the ChatBot...\n")
    dialog_eval = importlib.import_module("dialog-eval.code")
    evaluation_filenames = dataset.evaluation_files(**kwargs)

    config = dialog_eval.Config()
    config.train_source = evaluation_filenames["train_source"]
    config.test_source = evaluation_filenames["test_source"]
    config.test_target = evaluation_filenames["test_target"]
    config.test_responses = os.path.join(
        os.getcwd(), f"test-responses-{dataset.dataset_percentage}.txt"
    )
    config.text_vocab = evaluation_filenames["text_vocab"]
    config.vector_vocab = evaluation_filenames["vector_vocab"]
    config.lang = kwargs["lang"] if "lang" in kwargs else "en"

    for metric_name in config.metrics.keys():
        config.metrics[metric_name] = 1

    metrics = dialog_eval.Metrics(config)
    metrics.run()


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()

    logger.info("Building the dataset...\n")
    dataset = build_dataset(args)
    logger.info(
        f"The dataset contains {len(dataset.get_dataset_split(Split.train))} training files and {len(dataset.get_dataset_split(Split.test))} test files.\n"
    )

    kwargs = {}
    if args.open_subtitles:
        kwargs["lang"] = args.language

    dataset.downsize(args.dataset_fraction, **kwargs)
    training_filenames = dataset.get_dataset_split(Split.train)
    logger.info(
        f"Training the ChatBot with {dataset.dataset_percentage * 100:.2f}% of the data ({len(training_filenames)} files)...\n"
    )
    comparison_function = get_comparison_function(args)

    if not os.path.exists("databases"):
        os.makedirs("databases")

    if os.path.exists(os.path.join("databases", f"db-{args.name}.sqlite3")):
        os.remove(os.path.join("databases", f"db-{args.name}.sqlite3"))

    bot = ChatBot(
        args.name,
        preprocessors=["utils.preprocessor"],
        filters=["chatterbot.filters.get_recent_repeated_responses"],
        statement_comparison_function=comparison_function,
        storage_adapter="chatterbot.storage.SQLStorageAdapter",
        database_uri=f"sqlite:///databases/db-{args.name}.sqlite3",
        read_only=True,
    )

    trainer = ChatterBotCorpusTrainer(bot)

    start = time.time()
    trainer.train(*training_filenames)
    end = time.time()

    log_line()
    logger.info(f"Training finished! It took {end - start:.2f} seconds.\n")

    if args.run_test:
        start = time.time()
        run_against_test(bot, dataset)
        end = time.time()
        log_line()
        logger.info(f"Testing finished! It took {end - start:.2f} seconds.\n")
        test_filename = os.path.join(
            os.getcwd(), f"test-responses-{dataset.dataset_percentage}.txt"
        )
        logger.info(f'ChatBot responses have been saved to "{test_filename}" file.\n')
        if args.evaluate:
            start = time.time()
            run_evaluation(bot, dataset, **kwargs)
            end = time.time()
            log_line()
            logger.info(f"Evaluation finished! It took {end - start:.2f} seconds.\n")
            logger.info(
                f"Metrics have been saved to \"{os.path.join(os.getcwd(), 'metrics.txt')}\" file.\n"
            )
    else:
        enter_interactive_mode(bot)
