class ImageTitleCreator:
    """
    Given a two .txt file containing \n separated lines (synsets.txt and hyponyms.txt),
    [each lines contains some words that are comma separated]
    it generates two lists:
    - A list of synset titles
    - A list of hyponym titles
    """

    def __init__(self, file_path_synsets: str = "synsets.txt", file_path_hyponyms: str = "hyponyms.txt") -> None:
        self.synset_titles = []
        self.hyponym_titles = []
        self.synset_hyponym = {}
        self.create_titles(file_path_synsets, file_path_hyponyms)

    def create_titles(self, syn_path, hyp_path) -> None:
        """
        Creates a list of synset titles and a list of hyponym titles
        """
        with open(syn_path, "r") as f:
            synset_lines = f.readlines()
        with open(hyp_path, "r") as f:
            hyponym_lines = f.readlines()

        self.synset_titles = [line.strip() for line in synset_lines]
        self.hyponym_titles = [line.strip() for line in hyponym_lines]
        self.synset_hyponym = {synset: hyponym for synset, hyponym in zip(self.synset_titles, self.hyponym_titles)}

    def get_synset_titles(self) -> list:
        return self.synset_titles

    def get_hyponym_titles(self) -> list:
        return self.hyponym_titles

    def get_synset_hyponym(self) -> dict:
        return self.synset_hyponym


if __name__ == "__main__":
    itc = ImageTitleCreator()
    synset_hyponym = itc.get_synset_hyponym()
    from pprint import pprint

    pprint(synset_hyponym)
