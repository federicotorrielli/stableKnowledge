import os

from sentence_transformers import util, SentenceTransformer


class Evaluation:
    def __init__(self, generated_phrases_path_folder: str):
        self.generated_phrases_path = generated_phrases_path_folder
        self.folder_names = self.prepare_folder_names()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cosine_scores = self.compute_cosine_scores()

    def prepare_folder_names(self):
        temp = [folder.replace('_', ' ').replace('-', ',') for folder in os.listdir(self.generated_phrases_path)]
        returnvalue = {}
        for folder in temp:
            returnvalue[folder] = []
            with open(f"{self.generated_phrases_path}/{folder.replace(' ', '_').replace(',', '-')}/interrogations.txt",
                      "r") as f:
                returnvalue[folder] = f.readlines()
        return returnvalue

    def print_to_file(self) -> None:
        """
        Print the results to two files "cosine_scores.txt" and "normalized_scores.txt"
        for each folder, grouping the content of each cosine similarity and normalized similarity
        """
        for folder in self.cosine_scores:
            if len(self.cosine_scores[folder]) > 0:
                with open(
                        f"{self.generated_phrases_path}/{folder.replace(' ', '_').replace(',', '-')}/cosine_scores.txt",
                        "w") as f:
                    for score in self.cosine_scores[folder]:
                        f.write(f"{score}\n")
                    f.write(f"Mean: {sum(self.cosine_scores[folder]) / len(self.cosine_scores[folder])}\n")

    def compute_cosine_scores(self):
        """
        Compute the cosine similarity between the generated phrases and the folder name
        :return:
        """
        cosine_scores = {}
        for folder, phrases in self.folder_names.items():
            cosine_scores[folder] = []
            for i, p in enumerate(phrases):
                cosine_scores[folder].append(util.cos_sim(self.model.encode(p, convert_to_tensor=True),
                                                          self.model.encode(folder, convert_to_tensor=True)).item())
        return cosine_scores
