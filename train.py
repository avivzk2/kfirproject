
import re
import sys
import csv
import numpy as np

from nltk.stem import PorterStemmer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# Increases CSV field size limit:
maxInt = sys.maxsize
decrement = True
while decrement:
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True


stemmer = PorterStemmer()
word_tok = re.compile(r'\w+')


def tokenizer(text):
    """String word tokenizer with stemming"""
    # tokens = [w.lower() for w in word_tok.findall(text)]
    tokens = [stemmer.stem(w.lower()) for w in word_tok.findall(text)]
    return tokens


def read_csv(path):
    with open(path, 'rU', newline='') as fs:
        reader = csv.reader(x.replace('\0', '') for x in fs)
        rows = [r for r in reader]
    return rows


# Actual code starts here

data = read_csv('imdb.csv')

# Creates sentences from IMDB db:
sentences = []
for i, (text, _) in enumerate(data):
    tmp_sents = [()]
    tmp_sents = [(str(i), x.strip()) for x in re.split(r'\n|\.', text) if x]
    sentences.extend(tmp_sents)


# Returns list of tagged sentences:
tagged_sents = [TaggedDocument(words=tokenizer(s), tags=[p]) for p, s in sentences]


MAX_EPOCHS = 1
VEC_DIM = 200
WINDOW = 10
ALPHA = 0.025
MIN_ALPHA = 0.00025
OUTPUT = 'trained'

model = Doc2Vec(
    vector_size=VEC_DIM, window=WINDOW, alpha=ALPHA, min_alpha=MIN_ALPHA, min_count=1, dm=1,
)

model.build_vocab(tagged_sents)
for epoch in range(1, MAX_EPOCHS + 1):
    print('\rIteration {}/{}'.format(epoch, MAX_EPOCHS))
    model.train(tagged_sents, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.0002           # Decrease the learning rate
    model.min_alpha = model.alpha   # Fix the learning rate, no decay

# Saving model:
model.save(OUTPUT + '.model')



