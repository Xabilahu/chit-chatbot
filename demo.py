import os
import random
import sys

from chatterbot import ChatBot
from chatterbot.comparisons import (
    JaccardSimilarity,
    LevenshteinDistance,
    SpacySimilarity,
)
from chatterbot.conversation import Statement
from chatterbot.trainers import ChatterBotCorpusTrainer

from data import OpenSubtitles, PersonaChat, Split, WizardOfWikipedia

# Arguments defining name, comparison method, data and if is neccesary language
if "Spacy" in sys.argv[2]:
    bot = ChatBot(sys.argv[1], statement_comparison_function=SpacySimilarity)
elif "Jacca" in sys.argv[2]:
    bot = ChatBot(sys.argv[1], statement_comparison_function=JaccardSimilarity)
elif "Leve" in sys.argv[2]:
    bot = ChatBot(sys.argv[1], statement_comparison_function=LevenshteinDistance)

trainer = ChatterBotCorpusTrainer(bot)


if "Open" in sys.argv[3] and sys.argv[4] is not None:
    o = OpenSubtitles()
    filenames = o.get_dataset_split(Split.train, lang=sys.argv[4])
elif "Persona" in sys.argv[3]:
    p = PersonaChat()
    filenames = p.get_dataset_split(Split.train)
elif "Wizard" in sys.argv[3]:
    w = WizardOfWikipedia()
    filenames = w.get_dataset_split(Split.train)


idx = random.randint(0, len(filenames) - 10)
# trainer.train(*filenames[idx : idx + 10])
trainer.train(filenames)


def writefile(input, output):
    # os.makedirs(os.path.dirname("/results/"), exist_ok=True)
    with open(sys.argv[1] + ".txt", "a") as f:
        f.write(f'Input: "{input}"\n  Response: "{output}"\n')
        f.close()


while True:
    i = input()
    o = bot.get_response(i)
    print(o)
    writefile(i, o)
