import os

from sentence_transformers import util, SentenceTransformer


class Evaluation:
    def __init__(self, generated_phrases_path_folder: str):
        self.generated_phrases_path = generated_phrases_path_folder
        self.folder_names = self.prepare_folder_names()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cosine_scores = self.compute_cosine_scores()

    def prepare_folder_names(self):
        def process_folder_name(name):
            return name.replace('_', ' ').replace('-', ',')

        folder_names = {}
        for folder in os.listdir(self.generated_phrases_path):
            processed_folder_name = process_folder_name(folder)
            folder_names[processed_folder_name] = []
            with open(f"{self.generated_phrases_path}/{folder}/interrogations.txt", "r") as f:
                folder_names[processed_folder_name] = f.readlines()
        return folder_names

    def print_to_file(self) -> None:
        """
        Print the results to two files "cosine_scores.txt"
        for each folder, grouping the content of each cosine similarity
        """

        def process_folder_name(name):
            return name.replace(' ', '_').replace(',', '-')

        for folder, scores in self.cosine_scores.items():
            if scores:
                with open(f"{self.generated_phrases_path}/{process_folder_name(folder)}/cosine_scores.txt", "w") as f:
                    for score in scores:
                        f.write(f"{score}\n")
                    f.write(f"Mean: {sum(scores) / len(scores)}\n")

    def compute_cosine_scores(self):
        """
        Compute the cosine similarity between the generated phrases and the folder name
        """
        cosine_scores = {}
        for folder, phrases in self.folder_names.items():
            cosine_scores[folder] = []
            for i, p in enumerate(phrases):
                cosine_scores[folder].append(util.cos_sim(self.model.encode(p, convert_to_tensor=True),
                                                          self.model.encode(folder, convert_to_tensor=True)).item())
        return cosine_scores
