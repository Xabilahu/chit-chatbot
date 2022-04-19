import logging
import os
import sys
from enum import Enum
from typing import Callable, Dict, List

from chatterbot.conversation import Statement
from nltk.tokenize import wordpunct_tokenize


class Split(Enum):
    train = "train"
    test = "test"


# Class to redirect printing to sys.stdout to custom logger.
# (Check https://github.com/gunthercox/ChatterBot/blob/4ff8af28567ed446ae796d37c246bb6a14032fe7/chatterbot/utils.py#L93-L119)
class LoggerWriter:
    def __init__(self, logfunc: Callable) -> None:
        self.logfunc = logfunc
        self.prev = None

    def write(self, msg: str) -> None:
        if msg.endswith("100%") and msg != self.prev:
            self.prev = msg
            self.logfunc(f"{msg[1:]}\n")
        elif msg != self.prev:
            self.logfunc(f"{msg[1:]}\r")

    def flush(self) -> None:
        pass
        # self.logfunc("\n", extra={"simple": True})


# Taken from: https://stackoverflow.com/questions/34954373/disable-format-for-some-messages
class ConditionalFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "simple") and record.simple:
            return record.getMessage()
        else:
            return logging.Formatter.format(self, record)


def sanitize_yaml(text: str) -> str:
    # Unify quote characters into ' so that we can later quote the whole text with "
    # (avoid conflicts with YAML indicator symbols : and -)
    return text.translate(text.maketrans('"/\\', "'  "))


def preprocess(text: str) -> List[str]:
    return wordpunct_tokenize(sanitize_yaml(text))


def convert_chatterbot_filename(filename: str) -> str:
    basename = os.path.join(
        sys.exec_prefix,
        "lib",
        f"python{sys.version_info[0]}.{sys.version_info[1]}",
        "site-packages",
        "chatterbot_corpus",
        "data",
    )

    for path_segment in filename.split(".")[2:]:
        basename = os.path.join(basename, path_segment)

    return f"{basename}.yml"


def preprocessor(stmt: Statement) -> Statement:
    stmt.text = " ".join(preprocess(stmt.text.strip()))
    return stmt


def FlairSimilarity(a: Statement, b: Statement) -> float:
    raise NotImplementedError
