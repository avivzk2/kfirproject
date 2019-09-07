
import os
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


MAX_EPOCHS = 1
VEC_DIM = 200
WINDOW = 10
ALPHA = 0.025
MIN_ALPHA = 0.00025

data = read_csv('imdb.csv')
model = Doc2Vec.load('trained.model')


import random

for i in range(10):
    rid = random.randint(0, len(data))
    query = data[rid][0]
    vec = model.infer_vector(tokenizer(query))
    docs = model.docvecs.most_similar([vec], topn=5)

    print('=' * 100)
    print("QUERY: (id={})".format(rid))
    print(query)
    print('-' * 100)
    for j, doc in enumerate(docs):
        if str(rid) == doc[0]:
            continue
        print("RESULT {}".format(j))
        print(doc)
        print(data[int(doc[0])])
        print('-' * 100)

exit(0)



while True:

    print("=" * 100)
    # Loads review by index:
    query_id = input("Enter ID for a review as query [e.g. 178, ENTER to exit]: ")
    if not query_id:
        break
    query = data[int(query_id)][0]

    vec = model.infer_vector(tokenizer(query))
    docs = model.docvecs.most_similar([vec], topn=5)

    for doc in docs:
        print(doc)



