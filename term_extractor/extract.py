from collections import Counter

import nltk

from transformers_filter import TransFilter


def filter_synsets(synset_counter, common_nouns_synsets, n) -> list:
    most_common_synsets = [synset for synset, _ in synset_counter.most_common(n)]
    flush_set = set()
    for synset_list in common_nouns_synsets:
        for synset in synset_list:
            flush_set.add(synset)
    for synset in most_common_synsets:
        flush_set.add(synset)
    # Return most_common_synsets + flush_set
    return list(flush_set)


def retrieve_ogden_words() -> list:
    """
    Returns a list of words from the Ogden Basic English Dictionary
    ogden.txt is a one-line file containing all the words, split by commas
    """
    with open('ogden.txt', 'r') as f:
        words = f.read()
    return words.split(',')


def retrieve_common_nouns() -> list:
    """
    Returns a list of common nouns from the Common Nouns list
    common_nouns.txt is a multi-line file containing all the words
    """
    with open('common_nouns.txt', 'r') as f:
        words = f.read()
    return words.split('\n')


def synset_converter(words) -> list:
    """
    Converts words to synsets
    """
    return [nltk.corpus.wordnet.synsets(word) for word in words]


def count_synsets(synsets) -> Counter:
    """
    Counts the synsets
    """
    synset_counter = Counter()
    synset_counter.update(synsets)
    return synset_counter


def isolate_nouns(synsets) -> list:
    """
    Isolates nouns from synsets
    """
    final_synsets = set()
    for syn_list in synsets:
        if syn_list:
            for synset in syn_list:
                if synset.pos() == 'n':
                    final_synsets.add(synset)
    return list(final_synsets)


def get_best_synsets(n) -> list:
    """
    Returns the most frequent synsets
    """
    ogden_words = retrieve_ogden_words()
    common_nouns = retrieve_common_nouns()
    words = semcor.words()
    ogden_synsets = synset_converter(ogden_words)
    common_nouns_synsets = synset_converter(common_nouns)
    word_synsets = synset_converter(words)
    merged_synsets = ogden_synsets + word_synsets
    synsets = isolate_nouns(merged_synsets)
    synset_counter = count_synsets(synsets)
    return filter_synsets(synset_counter, common_nouns_synsets, n)


def get_most_frequent_hyponyms(most_frequent_synsets) -> dict:
    """
    Returns the most frequent hyponyms.
    Build a dictionary of the form {synset: best_hyponym}
    where the best_hyponym is the hyponym with the highest frequency
    and not too close with his hypernym
    """
    return get_best_hyponym(get_hyponyms(most_frequent_synsets))


def get_hyponyms(most_frequent_synsets) -> dict:
    return {synset: list(set([i for i in synset.closure(lambda s: s.hyponyms())])) for synset in most_frequent_synsets}


def get_best_hyponym(hyponyms: dict):
    return {synset: get_best_hyponym_from_list(hyponym_list, synset) for synset, hyponym_list in
            hyponyms.items()}


def get_best_hyponym_from_list(hyponym_list, synset) -> nltk.corpus.reader.wordnet.Synset:
    """
    Returns the best hyponym from a list of hyponyms.
    The best hyponym is the hyponym with a high frequency (in the top 10 of most frequent hyponyms)
    and not too close with his hypernym, with a path_similarity > 0.63.
    It is also required that the hypoynm cannot share a word with the hypernym.
    """
    for hyponym in hyponym_list:
        if hyponym.pos() == 'n' \
                and hyponym.path_similarity(synset) <= 0.63 \
                and is_in_top_10(hyponym, hyponym_list) \
                and not share_words(hyponym, synset) \
                and not is_common(hyponym):
            return hyponym
    for hyponym in hyponym_list:
        if hyponym.pos() == 'n' \
                and hyponym.path_similarity(synset) <= 0.63 \
                and not share_words(hyponym, synset) \
                and not is_common(hyponym):
            return hyponym
    return None


def is_in_top_10(hyponym, hyponym_list) -> bool:
    """
    Returns True if the hyponym is in the top 5 of most frequent hyponyms
    """
    hyponym_counter = Counter()
    hyponym_counter.update(hyponym_list)
    return hyponym in [hyponym for hyponym, _ in hyponym_counter.most_common(10)]


def share_words(hyponym, synset) -> bool:
    """
    Returns True if the hyponym and the synset share a word
    """
    for word in hyponym.lemma_names():
        for word2 in synset.lemma_names():
            if word in word2 or word2 in word:
                return True
    return False


def is_common(hyponym) -> bool:
    """
    Check if the hyponym's lexeme is a middle-concept
    """
    if len(synsets) == 0:
        # This should never happen, but just in case... (it happened)
        return False
    else:
        # Convert synsets in a list of their lexemes (words)
        synsets_lexemes = [synset.lemma_names() for synset in synsets]
        # Flatten the list
        synsets_lexemes = [item for sublist in synsets_lexemes for item in sublist]
        # Check if the hyponym's lexeme is in the list
        return hyponym.lemma_names()[0] in synsets_lexemes


def flush_to_file_hr(synsets, file_name) -> None:
    """
    Flushes the synsets, synonyms and hyponyms to a single file
    """
    for synset, hyponym in synsets.items():
        with open(file_name, 'a') as f:
            if synset and hyponym:
                f.write(f"Synset: {synset.name()} ({[str(lemma.name()) for lemma in synset.lemmas()][0]}) ||"
                        f" Hyponym: {hyponym.name()} || Synonyms: {synset.lemma_names()} || Hyponym's "
                        f"synonyms: {hyponym.lemma_names()}\n")


def flush_to_file_mr(synsets: dict, file_name: str, hyponym_file_name: str) -> None:
    """
    Flushes all the synset's lemmas (comma separated) to a file
    """
    flushed_synsets = set()
    flushed_hyponyms = set()
    for synset, hyponym in synsets.items():
        with open(file_name, 'a') as f:
            if synset and hyponym and synset not in flushed_synsets and synset not in flushed_hyponyms:
                lemmas = [str(lemma.name()).replace("_", " ") for lemma in synset.lemmas()]
                f.write(f"{','.join(lemmas)}\n")
                flushed_synsets.add(synset)
        with open(hyponym_file_name, 'a') as f:
            if synset and hyponym and hyponym not in flushed_synsets and hyponym not in flushed_hyponyms:
                lemmas = [str(lemma.name()).replace("_", " ") for lemma in hyponym.lemmas()]
                f.write(f"{','.join(lemmas)}\n")
                flushed_hyponyms.add(hyponym)


def keep_it_1000(file_name: str) -> None:
    """
    Keeps the file with 1000 lines
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()
    with open(file_name, 'w') as f:
        for line in lines[:1000]:
            f.write(line)


def main() -> None:
    """
    Main function
    """
    most_frequent_synsets = get_best_synsets(300)  # 300 is totally arbitrary, it WON'T be the final number.

    # Now we want to get the most frequent hyponyms of the most frequent synsets
    unfiltered_synsets = get_most_frequent_hyponyms(most_frequent_synsets)

    # Process the synsets to remove the ones that are not common english
    tf = TransFilter(unfiltered_synsets)
    tf.batch_processing()
    filtered_synsets = tf.get_filtered_synsets()

    # Flush everything to a file
    flush_to_file_mr(filtered_synsets, 'synsets.txt', 'hyponyms.txt')
    # Keep only the first 1000 lines from the file
    keep_it_1000('synsets.txt')
    keep_it_1000('hyponyms.txt')


if __name__ == '__main__':
    semcor = nltk.corpus.semcor
    synsets = []
    main()
