import csv
import gzip
import json
import os
import random
import re
import shutil
import sys
import tarfile
import zipfile
from abc import ABCMeta, abstractmethod
from itertools import chain
from typing import Dict, List, Tuple
from xml.etree import ElementTree

import urllib3
import yake
import yaml
from tqdm import tqdm

from utils import Split, preprocess, sanitize_yaml

FASTTEXT_DOWNLOAD_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{}.300.vec.gz"
)


class DataWrapper(metaclass=ABCMeta):
    def __init__(self, name: str, url: str, *langs: str) -> None:
        self.name = name
        self.url = url
        self.languages = set(langs)
        self.filenames: List[Tuple[Split, str]] = []
        self.dataset_percentage = 1.0
        self.data_path = os.path.join(
            sys.exec_prefix,
            "lib",
            f"python{sys.version_info[0]}.{sys.version_info[1]}",
            "site-packages",
            "chatterbot_corpus",
            "data",
            self.name,
        )

    def _append_evaluation(self, infile: str, outfile: str, index: Tuple[int]) -> None:
        infp = open(infile, "r")
        contents = yaml.safe_load(infp)
        with open(outfile, "a") as f:
            for conversation in contents["conversations"]:
                for utterance in conversation[index[0] : index[1]]:
                    sentence = (
                        utterance[1:-1]
                        if utterance.startswith('"') and utterance.endswith('"')
                        else utterance
                    )
                    f.write(f"{sentence}\n")
        infp.close()

    def evaluation_files(self, **kwargs) -> Dict[str, str]:
        if self.filenames == []:
            self.get_dataset_split(Split.train, **kwargs)

        lang = "en" if "lang" not in kwargs else kwargs["lang"]
        embedding_filename = os.path.join(
            os.path.dirname(self.data_path), f"cc.{lang}.vec"
        )
        if not os.path.exists(embedding_filename):
            self._download_data(
                f"{embedding_filename}.gz", url=FASTTEXT_DOWNLOAD_URL.format(lang)
            )
            with open(embedding_filename, "wb") as outfp, gzip.open(
                f"{embedding_filename}.gz"
            ) as infp:
                while True:
                    chunk = infp.read(8192)
                    if not chunk:
                        break
                    outfp.write(chunk)
            os.remove(f"{embedding_filename}.gz")

        base_path = (
            os.path.join(self.data_path, kwargs["lang"])
            if "lang" in kwargs
            else self.data_path
        )
        evaluation_filenames = dict()
        for key, filename in [
            ("train_source", "train-source-{}.txt"),
            ("text_vocab", "vocab-{}.txt"),
            ("vector_vocab", "vocab-{}.npy"),
            ("test_source", "test-source-{}.txt"),
            ("test_target", "test-target-{}.txt"),
        ]:
            evaluation_filenames[key] = os.path.join(
                base_path, filename.format(self.dataset_percentage)
            )

        for split, filename in self.filenames:
            full_filename = os.path.join(base_path, f"{filename}.yml")
            if split == Split.train:
                self._append_evaluation(
                    full_filename,
                    os.path.join(base_path, evaluation_filenames["train_source"]),
                    (0, -1),
                )
            elif split == Split.test:
                self._append_evaluation(
                    full_filename,
                    os.path.join(base_path, evaluation_filenames["test_source"]),
                    (0, -1),
                )
                self._append_evaluation(
                    full_filename,
                    os.path.join(base_path, evaluation_filenames["test_target"]),
                    (1, None),
                )

        # Now we generate the vocab file
        vocab = set()
        with open(evaluation_filenames["text_vocab"], "w") as outfp, open(
            evaluation_filenames["train_source"], "r"
        ) as infp:
            while True:
                line = infp.readline()
                if not line:
                    break
                words = line.strip().split(" ")
                vocab.update(words)
                outfp.write("\n".join(words))
                outfp.write("\n")

        with open(embedding_filename, "r") as infp, open(
            evaluation_filenames["vector_vocab"], "w"
        ) as outfp:
            while True:
                line = infp.readline()
                if not line:
                    break
                tokens = line.strip().split()
                if tokens[0] not in vocab:
                    continue
                if len(tokens) == 301:
                    outfp.write(line)
                elif tokens[1] == "»":
                    outfp.write(f"{tokens[0]} {' '.join(tokens[2:])}\n")

        return evaluation_filenames

    def downsize(self, percentage: float, **kwargs) -> None:
        if percentage > 1.0 or percentage <= 0.0:
            raise ValueError("Downsize percentage must be in the range (0,1].")

        if self.filenames == []:
            self._call_data_preparation(self.data_path, **kwargs)

        self.dataset_percentage *= percentage
        train_size = int(
            len(list(filter(lambda x: x[0] == Split.train, self.filenames)))
            * percentage
        )
        test_size = int(
            len(list(filter(lambda x: x[0] == Split.test, self.filenames))) * percentage
        )

        random.shuffle(self.filenames)
        train_count, test_count = 0, 0
        new_filenames = []
        for split, filename in self.filenames:
            if split == Split.train and train_count < train_size:
                new_filenames.append((split, filename))
                train_count += 1
            elif split == Split.test and test_count < test_size:
                new_filenames.append((split, filename))
                test_count += 1

        self.filenames = new_filenames

    @abstractmethod
    def prepare_data(self, **kwargs) -> None:
        raise NotImplementedError

    def _call_data_preparation(self, data_path: str, **kwargs) -> None:
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            self.prepare_data(**kwargs)
        else:
            if len(self.filenames) == 0:
                with open(os.path.join(data_path, "split_mapping.csv"), "r") as f:
                    csvreader = csv.reader(f)
                    _ = next(csvreader)  # Skip header
                    for row in csvreader:
                        self.filenames.append(
                            (Split.train if row[0] == "train" else Split.test, row[1])
                        )

            present_files = set(os.listdir(data_path))
            for _, file in self.filenames:
                if f"{file}.yml" not in present_files:
                    self.prepare_data(**kwargs)
                    break

    def get_dataset_split(self, split: Split, **kwargs) -> List[str]:
        self._call_data_preparation(self.data_path, **kwargs)

        return [
            ".".join(["chatterbot", "corpus", self.name, filename])
            for s, filename in self.filenames
            if s == split
        ]

    def _download_data(
        self, filename: str, chunk_size: int = 8192, url: str = None
    ) -> None:
        http = urllib3.PoolManager()
        r = http.request("GET", self.url if url is None else url, preload_content=False)
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

    def available_languages(self) -> List[str]:
        return self.languages


class PersonaChat(DataWrapper):
    def __init__(self) -> None:
        super(PersonaChat, self).__init__(
            "Persona-Chat",
            "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json",
            "en",
        )

    def prepare_data(self, **kwargs) -> None:
        filename = os.path.join(self.data_path, "personachat_self_original.json")
        self._download_data(filename)

        keyword_extractor = yake.KeywordExtractor(lan="en", n=1, windowsSize=3, top=1)
        with open(filename, "r") as f:
            contents = json.load(f)

        with open(os.path.join(self.data_path, "split_mapping.csv"), "w") as csvfile:
            csvfile.write("split,filename\n")
            train_target, counter = (
                int((len(contents["train"]) + len(contents["valid"])) * 0.8),
                1,
            )
            for dialog in chain(contents["train"], contents["valid"]):
                idx, split = counter, Split.train
                if counter > train_target:
                    idx -= train_target
                    split = Split.test

                c_filename = os.path.join(self.data_path, f"{split.value}-{idx}.yml")
                self.filenames.append(
                    (split, os.path.splitext(os.path.basename(c_filename))[0])
                )
                csvfile.write(
                    f"{split.value},{os.path.splitext(os.path.basename(c_filename))[0]}\n"
                )
                with open(c_filename, "w") as outfile:
                    keywords = set()
                    for p in dialog["personality"]:
                        keyword_candidates = keyword_extractor.extract_keywords(p)
                        if len(keyword_candidates) > 0:
                            keywords.add(keyword_candidates[0][0])
                    outfile.write("categories:\n")
                    for kw in keywords:
                        outfile.write(f"- {kw}\n")
                    outfile.write("conversations:\n")
                    first = True
                    for utterance in dialog["utterances"][-1]["history"]:
                        sentence = " ".join(preprocess(utterance))
                        if first:
                            outfile.write(f'- - "{sentence}"\n')
                            first = False
                        else:
                            outfile.write(f'  - "{sentence}"\n')
                counter += 1
        del contents
        os.remove(filename)


class WizardOfWikipedia(DataWrapper):
    def __init__(self) -> None:
        super(WizardOfWikipedia, self).__init__(
            "Wizard-Of-Wikipedia",
            "http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz",
            "en",
        )

    def prepare_data(self, **kwargs) -> None:
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
                    topic_splits[sanitize_yaml(topic_name)] = split
            del contents

        with open(os.path.join(self.data_path, "data.json"), "r") as infile, open(
            os.path.join(self.data_path, "split_mapping.csv"), "w"
        ) as csvfile:
            contents = json.load(infile)
            csvfile.write("split,filename\n")
            for instance in contents:
                topic_name = sanitize_yaml(instance["chosen_topic"])
                if topic_name not in topic_splits:
                    continue
                topic_file = os.path.join(
                    self.data_path,
                    f"{'-'.join(topic_name.translate(topic_name.maketrans('.,', '- ')).split(' '))}.yml",
                )
                exists = os.path.exists(topic_file)
                with open(topic_file, "a") as outfile:
                    if not exists:
                        self.filenames.append(
                            (
                                topic_splits[topic_name],
                                os.path.splitext(os.path.basename(topic_file))[0],
                            )
                        )
                        csvfile.write(
                            f"{topic_splits[topic_name].value},{os.path.splitext(os.path.basename(topic_file))[0]}\n"
                        )
                        outfile.write(
                            f'categories:\n- "{topic_name}"\nconversations:\n'
                        )

                    first_sentence = True
                    for utterance in instance["dialog"]:
                        sentence = " ".join(preprocess(utterance["text"]))
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
    def __init__(self) -> None:
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

    def __parse_xml(self, filename: str) -> Tuple[List[str], List[str]]:
        try:
            root = ElementTree.parse(filename).getroot()
            dialogue = []
            for node in root.findall("s"):
                for child in node.getiterator():
                    if child.text is not None and child.text.strip() != "":
                        dialogue.append(child.text.strip())
                        break
                    if child.tail is not None and child.tail.strip() != "":
                        dialogue.append(child.tail.strip())
                        break

            categories = root.find("meta/source/genre")
            return (
                dialogue,
                ["subtitles"] if categories is None else categories.text.split(","),
            )
        except ElementTree.ParseError:
            return None, None

    def evaluation_files(self, **kwargs) -> List[str]:
        if "lang" not in kwargs:
            raise ValueError(
                f"Please provide an ISO 639-1 language code, by passing the keyword argument 'lang'. Accepted languages are: {', '.join(self.languages)}"
            )

        return super(OpenSubtitles, self).evaluation_files(**kwargs)

    def prepare_data(self, **kwargs) -> None:
        if "lang" not in kwargs:
            raise ValueError(
                f"Please provide an ISO 639-1 language code, by passing the keyword argument 'lang'. Accepted languages are: {', '.join(self.languages)}"
            )

        formatted_url = self.url.format(kwargs["lang"])
        file_basename = re.search(r"/(([^/]+)\.zip)$", formatted_url, re.M).group(1)
        filename = os.path.join(self.data_path, file_basename)
        self._download_data(filename, url=formatted_url)

        with zipfile.ZipFile(filename, "r") as zip_file:
            filenames = zip_file.namelist()
            extracted_filenames = list(filter(lambda x: x.endswith("xml"), filenames))
            zip_file.extractall(self.data_path)

        train_size = int(len(extracted_filenames) * 0.85)
        counter = 1
        with open(
            os.path.join(self.data_path, kwargs["lang"], "split_mapping.csv"), "w"
        ) as csvfile:
            csvfile.write("split,filename\n")
            for xml_filename in extracted_filenames:
                dialogue, categories = self.__parse_xml(
                    os.path.join(self.data_path, xml_filename)
                )
                if dialogue is not None and categories is not None:
                    if counter > train_size:
                        yml_filename = os.path.join(
                            self.data_path,
                            kwargs["lang"],
                            f"test-{counter - train_size}.yml",
                        )
                        self.filenames.append(
                            (
                                Split.test,
                                os.path.splitext(os.path.basename(yml_filename))[0],
                            )
                        )
                    else:
                        yml_filename = os.path.join(
                            self.data_path, kwargs["lang"], f"train-{counter}.yml"
                        )
                        self.filenames.append(
                            (
                                Split.train,
                                os.path.splitext(os.path.basename(yml_filename))[0],
                            )
                        )

                    csvfile.write(
                        f"{self.filenames[-1][0].value},{self.filenames[-1][1]}\n"
                    )
                    with open(yml_filename, "w") as f:
                        f.write("categories:\n")
                        for category in categories:
                            f.write(f'- "{sanitize_yaml(category)}"\n')
                        f.write("conversations:\n")
                        prev_sentence = " ".join(preprocess(dialogue[0]))
                        for i in range(1, len(dialogue)):
                            sentence = " ".join(preprocess(dialogue[i]))
                            f.write(f'- - "{prev_sentence}"\n  - "{sentence}"\n')
                            prev_sentence = sentence
                counter += 1

        os.remove(filename)
        for filename in filenames:
            file_path = os.path.join(self.data_path, filename)
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    shutil.rmtree(file_path, ignore_errors=True)

    def get_dataset_split(self, split: Split, **kwargs) -> List[str]:
        if "lang" not in kwargs:
            raise ValueError(
                f"Please provide an ISO 639-1 language code, by passing the keyword argument 'lang'. Accepted languages are: {', '.join(self.languages)}"
            )

        base_path = os.path.join(self.data_path, kwargs["lang"])
        self._call_data_preparation(base_path, **kwargs)

        return [
            ".".join(["chatterbot", "corpus", self.name, kwargs["lang"], filename])
            for s, filename in self.filenames
            if s == split
        ]
