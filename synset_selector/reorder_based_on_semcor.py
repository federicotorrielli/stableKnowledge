import random
from collections import Counter
from typing import List, Tuple, Dict

from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset as syn


# This code is so bad that I don't even want to look at it anymore
# So please don't judge me for it: it does its job, and that's all that matters

def create_dict(file1: str, file2: str) -> Dict[syn, syn]:
    synset_dict = {}

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    for line1, line2 in zip(lines1, lines2):
        synset1 = wn.synset(line1.split("'):")[0].split("('")[1])
        synset2 = wn.synset(line2.split("'):")[0].split("('")[1])

        synset_dict[synset1] = synset2

    return synset_dict


def reorder_based_on_semcor_frequency(synset_dict: Dict[syn, syn]) -> List[Tuple[syn, syn]]:
    synsets_list = []
    for sent in semcor.sents():
        for word in sent:
            sent_synsets = wn.synsets(word)
            synsets_list.extend(sent_synsets)

    counter = Counter(synsets_list)
    sorted_synsets = [(synset, synset_dict[synset]) for synset in sorted(counter, key=counter.get, reverse=True) if
                      synset in synset_dict]
    return sorted_synsets


def write_synset_to_file(synset, related_synset, f1, f2):
    sln = ", ".join(s.replace("_", " ") for s in synset.lemma_names())
    rsln = ", ".join(s.replace("_", " ") for s in related_synset.lemma_names())
    f1.write(f"{synset}:{sln} | Definition: {synset.definition()}\n")
    f2.write(f"{related_synset}:{rsln} | Definition: {related_synset.definition()}\n")


def create_dataset(sorted_synsets):
    with open("../dataset_extractor/dataset.txt", 'w') as f:
        # Keep only the first 500 synsets
        already_used_lemmas = set()
        i = 0
        for synset, related_synset in sorted_synsets:
            if synset.lemma_names()[0] not in already_used_lemmas and related_synset.lemma_names()[
                0] not in already_used_lemmas:
                write_synset_to_file(synset, related_synset, f, f)
                already_used_lemmas.add(synset.lemma_names()[0])
                already_used_lemmas.add(related_synset.lemma_names()[0])
                i += 1
                if i == 250:
                    break
        f.close()

    # open the file and read in the lines
    with open('../dataset_extractor/dataset.txt', 'r') as f:
        lines = f.readlines()

    # shuffle the lines using the random module
    random.shuffle(lines)

    # write the shuffled lines back to the file
    with open('../dataset_extractor/dataset.txt', 'w') as f:
        for line in lines:
            f.write(line)
        f.close()


def main() -> None:
    """
    Reorder the synsets based on their frequency in the SemCor corpus.
    :return: None
    """
    # create a dictionary mapping synsets to their corresponding synsets in the SemCor corpus
    synset_dict = create_dict("synsets_with_glosses.txt", "hyponyms_with_glosses.txt")

    # reorder the synsets based on their frequency in the SemCor corpus
    sorted_synsets = reorder_based_on_semcor_frequency(synset_dict)
    already_used_lemmas = set()
    with open("../dataset_extractor/synsets_with_glosses.txt", 'a') as f1, open(
            "../dataset_extractor/hyponyms_with_glosses.txt", 'a') as f2:
        for synset, related_synset in sorted_synsets:
            if synset.lemmas()[0].name() not in already_used_lemmas and related_synset.lemmas()[
                0].name() not in already_used_lemmas:
                write_synset_to_file(synset, related_synset, f1, f2)
                already_used_lemmas.add(synset.lemmas()[0].name())
                already_used_lemmas.add(related_synset.lemmas()[0].name())
        f1.close()
        f2.close()

    create_dataset(sorted_synsets)


if __name__ == "__main__":
    main()
