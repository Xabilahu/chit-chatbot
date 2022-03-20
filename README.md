# chatbot
Final project for the NLP Applications I HAP/LAP Masters course

## Project description

While state-of-the-art chatbot systems use Transformers such as [GPT-2](https://arxiv.org/pdf/2004.12752.pdf), for this project you can use [Chatterbot](https://chatterbot.readthedocs.io/en/stable/), a very simple machine learning-based chatbot. While easy to use, Chatterbot provides only a tiny little amount of training data. Thus, it would be interesting to leverage larger corpora to train Chatterbot for a language(s) of your choice. 

Summarizing, this may involve the following: 

1. Create a virtual environment in your personal machine to install chatterbot and chatterbot-corpus.
2. Download your monolingual untokenized raw files for your chosen language [here](https://opus.nlpl.eu/OpenSubtitles-v2018.php).
3. Format the corpus to the YAML format used in Chatterbot and include it in the appropriate language directory ([instructions](https://github.com/gunthercox/chatterbot-corpus#create-your-own-corpus-training-data))

   __NOTE__: formatting the data requires saving it into YAML format, using a YAML parser, otherwise Chatterbot will output many parsing errors.
   
4. train the model.
5. chat with the Chatterbot performing a qualitative analysis of the dialogue, for example, by comparing the dialogues performed by the models trained with the tiny corpus available in Chatterbot or with the OpenSubtitles corpus.
6. __OPTIONAL__: Split the training data into train and test sets in the format required by the evaluation toolkit [dialog-eval](https://github.com/ricsinaruto/dialog-eval), re-train and evaluate the obtained model on the newly created testset.
