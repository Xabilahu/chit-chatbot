import csv
import json
import os
import sys
import tarfile
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import List, Tuple

import urllib3
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm


class Split(Enum):
    train = "train"
    test = "test"


class DataWrapper(metaclass=ABCMeta):
    def __init__(self, name: str, url: str, *langs: str):
        self.name = name
        self.url = url
        self.languages = set(langs)
        self.filenames: List[Tuple[Split, str]] = []
        self.data_path = os.path.join(
            sys.exec_prefix,
            "lib",
            f"python{sys.version_info[0]}.{sys.version_info[1]}",
            "site-packages",
            "chatterbot_corpus",
            "data",
            self.name,
        )

    def preprocess(self, text: str):
        # Unify quote characters into ' so that we can later quote the whole text with "
        # (avoid conflicts with YAML indicator symbols : and -)
        return wordpunct_tokenize(text.translate(text.maketrans('"', "'")))

    @abstractmethod
    def prepare_data(self):
        raise NotImplementedError

    def get_dataset_split(self, split: Split):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            self.prepare_data()
        else:
            if len(self.filenames) == 0:
                with open(os.path.join(self.data_path, "split_mapping.csv"), "r") as f:
                    csvreader = csv.reader(f)
                    _ = next(csvreader)  # Skip header
                    for row in csvreader:
                        self.filenames.append(
                            (Split.train if row[0] == "train" else Split.test, row[1])
                        )

            present_files = set(os.listdir(self.data_path))
            for _, file in self.filenames:
                if file.endswith("yaml") and file not in present_files:
                    self.prepare_data()
                    break

        return [
            ".".join(["chatterbot", "corpus", self.name, filename])
            for s, filename in self.filenames
            if s == split
        ]

    def _download_data(self, filename: str, chunk_size: int = 8192):
        http = urllib3.PoolManager()
        r = http.request("GET", self.url, preload_content=False)
        with tqdm(
            total=int(r.headers["Content-Length"]),
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            with open(filename, "wb") as f:
                while True:
                    data = r.read(chunk_size)
                    if not data:
                        break
                    f.write(data)
                    pbar.update(len(data))
        r.release_conn()

    def available_languages(self):
        return self.languages


class PersonaChat(DataWrapper):
    def __init__(self):
        super(PersonaChat, self).__init__(
            "Persona-Chat",
            "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json",
            "en",
        )

    def prepare_data(self):
        pass  # TODO


class WizardOfWikipedia(DataWrapper):
    def __init__(self):
        super(WizardOfWikipedia, self).__init__(
            "Wizard-Of-Wikipedia",
            "http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz",
            "en",
        )

    def prepare_data(self):
        filename = os.path.join(self.data_path, "wizard_of_wikipedia.tgz")
        self._download_data(filename)

        tar_file = tarfile.open(filename)
        tar_file.extractall(self.data_path)
        extracted_filenames = tar_file.getmembers()
        tar_file.close()

        topic_splits = dict()
        with open(os.path.join(self.data_path, "topic_splits.json"), "r") as f:
            contents = json.load(f)
            for split in Split:
                for topic_name in contents[split.value]:
                    topic_splits[topic_name] = split
            del contents

        with open(os.path.join(self.data_path, "data.json"), "r") as infile, open(
            os.path.join(self.data_path, "split_mapping.csv"), "w"
        ) as csvfile:
            contents = json.load(infile)
            csvfile.write("split,filename\n")
            for instance in contents:
                if instance["chosen_topic"] not in topic_splits:
                    continue
                topic_file = os.path.join(
                    self.data_path,
                    f"{'-'.join(instance['chosen_topic'].split(' '))}.yaml",
                )
                exists = os.path.exists(topic_file)
                with open(topic_file, "a") as outfile:
                    if not exists:
                        self.filenames.append(
                            (
                                topic_splits[instance["chosen_topic"]],
                                os.path.basename(topic_file),
                            )
                        )
                        csvfile.write(
                            f"{topic_splits[instance['chosen_topic']].value},{os.path.basename(topic_file)}\n"
                        )
                        outfile.write(
                            f"categories:\n- {instance['chosen_topic']}\nconversations:\n"
                        )

                    first_sentence = True
                    for utterance in instance["dialog"]:
                        sentence = " ".join(self.preprocess(utterance["text"]))
                        if first_sentence:
                            outfile.write(f'- - "{sentence}"\n')
                            first_sentence = False
                        else:
                            outfile.write(f'  - "{sentence}"\n')
            del contents

        os.remove(filename)
        for filename in extracted_filenames:
            os.remove(os.path.join(self.data_path, filename.name))


class OpenSubtitles(DataWrapper):
    def __init__(self):
        super(OpenSubtitles, self).__init__(
            "Open-Subtitles",
            "https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/raw/{}.zip",
            "af",
            "ar",
            "bg",
            "bn",
            "br",
            "bs",
            "ca",
            "cs",
            "da",
            "de",
            "el",
            "en",
            "eo",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fr",
            "gl",
            "he",
            "hi",
            "hr",
            "hu",
            "hy",
            "id",
            "is",
            "it",
            "ja",
            "ka",
            "kk",
            "ko",
            "lt",
            "lv",
            "mk",
            "ml",
            "ms",
            "nl",
            "no",
            "pl",
            "pt",
            "pt_br",
            "ro",
            "ru",
            "si",
            "sk",
            "sl",
            "sq",
            "sr",
            "sv",
            "ta",
            "te",
            "th",
            "tl",
            "tr",
            "uk",
            "ur",
            "vi",
            "ze_en",
            "ze_zh",
            "zh_cn",
            "zh_tw",
        )
        self.filename_template = "{}.yaml"

    def prepare_data(self):
        pass  # TODO

    def get_language(self, lang_id: str):
        if lang_id not in self.languages:
            raise ValueError(
                f"Unsupported language type {lang_id}. Accepted languages are: {' '.join(self.languages)}"
            )

        # TODO: download the file corresponding to the given lang_id and prepare it
