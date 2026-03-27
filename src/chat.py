# nltk -> Natural Language Toolkit, a Python library for processing human language.
# json -> To read or write JSON files (commonly used to store chatbot intents or text data).
# random -> For random selections (like picking a random response).
# numpy(np) -> For numerical operations, arrays, and computations.

# PorterStemmer -> A tool to reduce words to their root form(stemming). Example: running -> run, happily -> happi
# word_tokenize -> A function to split a sentence into words or tokens. Example: "Hello World!" -> ["Hello", "world", "!"]

# [stemmer.stem(word) for word in words] -> List comprehension: For each word in the tokenized list, apply the stemmer. Example: ["running", "jumps", "happily"] -> ["run", "jump", "happi"]
# preprocess(sentence): Return the stemmed tokens. The function outputs a list of simplified words. preprocess("I am running happily!") -> Output: ['i', 'am', 'run', 'happi', '!']

import nltk
import json
import random
import numpy as np

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

# Return the stemmed tokens. The function outputs a list of simplified words. 
def preprocess(sentence):
    words = word_tokenize(sentence.lower())
    return [stemmer.stem(word) for word in words]

# The function will convert a sentence into a bag-of words vector (Numbers). Example: tokenized_sentence = ["Hello", "you"], words = ["hello", "bye", "thanks", "you"] -> bag_of_words(tokenized_sentence, words) # Output: [1.0, 0.0, 0.0, 1.0]
def bag_of_words(tokenized_sentence, words):
    sentence_words = [stemmer.stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)

    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    
    return bag

# Prepare Training Data
with open("../data/intents.json") as file:
    data = json.load(file)

all_words = []
tags = []
xy = [] # (words, tag)

for intent in data["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for pattern in intent["patterns"]:
        w = preprocess(pattern)
        all_words.extend(w)
        xy.append((w, tag))

all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for(pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

#####
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

def chatbot():
    print("Chatbot is running! (type 'quit' to stop)")

    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        
        tokens = preprocess(sentence)
        bag = bag_of_words(tokens, all_words)

        prediction = model.predict([bag])[0]
        tag = tags[prediction]

        for intent in data["intents"]:
            if intent["tag"] == tag:
                print("Bot:", random.choice(intent["responses"]))

chatbot()