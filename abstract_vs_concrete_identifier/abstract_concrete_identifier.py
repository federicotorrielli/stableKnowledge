import numpy as np
import spacy
from nltk.corpus import wordnet as wn
from sklearn.linear_model import LogisticRegression

# download en_core_web_md
nlp = spacy.load("en_core_web_md")

with open("dataset.txt", 'r') as f:
    synsets = f.readlines()
    f.close()

for i in range(len(synsets)):
    synsets[i] = wn.synset(synsets[i].split("('")[1].split("')")[0])

classes = ['concrete', 'abstract']
train_set = [['spoon', 'fork', 'knife', 'table', 'chair', 'bed', 'car', 'bus', 'train', 'plane', 'computer', 'phone',
              'apple', 'owl', 'house', 'tree', 'flower', 'dog', 'cat', 'bird'],
             ['agony', 'anguish', 'anxiety', 'apprehension', 'awe', 'beauty', 'boredom', 'calm', 'contentment',
              'knowledge', 'process', 'love', 'friendship', 'democracy', 'set', 'morning', 'use', 'kernel', 'soul']]

X = np.stack([list(nlp(w))[0].vector for part in train_set for w in part])
y = [label for label, part in enumerate(train_set) for _ in part]
classifier = LogisticRegression(C=0.1, class_weight='balanced').fit(X, y)

for synset in synsets:
    synset_name = synset.lemma_names()[0]
    synset_vector = list(nlp(synset_name))[0].vector
    synset_class = classifier.predict([synset_vector])[0]
    # Write on a file called 'dataset_with_classes.txt' the synset and its class
    with open("dataset_with_classes.txt", 'a') as f:
        f.write(f"{synset}:{','.join(synset.lemma_names())} --> {classes[synset_class]}\n")
        f.close()

with open('../json_analyzer/super_annotator.txt', 'r') as f:
    super_annotator = f.readlines()
    f.close()

with open("super_annotator_concrete.txt", 'a') as f:
    for line in super_annotator:
        synset = wn.synset(line.split("('")[1].split("')")[0])
        synset_name = synset.lemma_names()[0]
        synset_vector = list(nlp(synset_name))[0].vector
        synset_class = classifier.predict([synset_vector])[0]
        if synset_class == 0:
            f.write(line)
