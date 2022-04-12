import random

from chatterbot import ChatBot
from chatterbot.conversation import Statement
from chatterbot.trainers import ChatterBotCorpusTrainer

from data import OpenSubtitles, Split

# from chatterbot.comparisons import SpacySimilarity


bot = ChatBot("spanish-subtitles")  # , statement_comparison_function=SpacySimilarity)
trainer = ChatterBotCorpusTrainer(bot)

o = OpenSubtitles()
filenames = o.get_dataset_split(Split.train, lang="es")

idx = random.randint(0, len(filenames) - 10)
trainer.train(*filenames[idx : idx + 10])

while True:
    print(bot.get_response(input()))
