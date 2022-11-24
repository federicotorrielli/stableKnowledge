from sentence_transformers import SentenceTransformer, util
import os


class Evaluation:
    def __init__(self, original_words_path_txt: str, generated_phrases_path_folder: str):
        self.original_words_path_txt = original_words_path_txt
        self.generated_phrases_path = generated_phrases_path_folder
        self.original_words = self.load_original_words()
        gp_1, gp_2, gp_3, gp_4, gp_5 = self.load_generated_phrases()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_original = self.model.encode(self.original_words, convert_to_tensor=True)
        emb_1, emb_2, emb_3, emb_4, emb_5 = self.load_embeddings(gp_1, gp_2, gp_3, gp_4, gp_5)
        self.cosine_scores = self.compute_cosine_scores(emb_1, emb_2, emb_3, emb_4, emb_5)
        self.normalized_scores = self.compute_normalized_scores()

    def compute_normalized_scores(self):
        """
        Normalize the cosine scores from 1 to 10
        :return:
        """
        normalized_scores = []
        for i in range(5):
            normalized_scores.append(self.cosine_scores[i] * 9 + 1)
        return normalized_scores

    def load_original_words(self) -> list:
        with open(self.original_words_path_txt, "r") as f:
            original_words = f.readlines()
        return original_words

    def load_generated_phrases(self) -> (list, list, list, list, list):
        # For every folder in the generated_phrases_path_folder
        # read the file 'interrogations.txt' and put in each list
        # the first, second, third, fourth and fifth phrase in the file

        gp_1, gp_2, gp_3, gp_4, gp_5 = [], [], [], [], []
        for folder in os.listdir(self.generated_phrases_path):
            with open(f"{self.generated_phrases_path}/{folder}/interrogations.txt", "r") as f:
                lines = f.readlines()
                gp_1.append(lines[0].strip())
                gp_2.append(lines[1].strip())
                gp_3.append(lines[2].strip())
                gp_4.append(lines[3].strip())
                gp_5.append(lines[4].strip())
        return gp_1, gp_2, gp_3, gp_4, gp_5

    def print_to_file(self) -> None:
        """
        Print the results to two files "cosine_scores.txt" and "normalized_scores.txt"
        for each folder, grouping the content of each cosine similarity and normalized similarity
        """
        for folder in os.listdir(self.generated_phrases_path):
            with open(f"{self.generated_phrases_path}/{folder}/cosine_scores.txt", "w") as f:
                for i in range(5):
                    f.write(f"{self.cosine_scores[i][1][1]}\n")
            with open(f"{self.generated_phrases_path}/{folder}/normalized_scores.txt", "w") as f:
                for i in range(5):
                    f.write(f"{self.normalized_scores[i][1][1]}\n")

    def load_embeddings(self, gp_1, gp_2, gp_3, gp_4, gp_5):
        embeddings1 = self.model.encode(gp_1, convert_to_tensor=True)
        embeddings2 = self.model.encode(gp_2, convert_to_tensor=True)
        embeddings3 = self.model.encode(gp_3, convert_to_tensor=True)
        embeddings4 = self.model.encode(gp_4, convert_to_tensor=True)
        embeddings5 = self.model.encode(gp_5, convert_to_tensor=True)
        return embeddings1, embeddings2, embeddings3, embeddings4, embeddings5

    def compute_cosine_scores(self, emb_1, emb_2, emb_3, emb_4, emb_5):
        return [util.pytorch_cos_sim(emb_1, self.embeddings_original),
                util.pytorch_cos_sim(emb_2, self.embeddings_original),
                util.pytorch_cos_sim(emb_3, self.embeddings_original),
                util.pytorch_cos_sim(emb_4, self.embeddings_original),
                util.pytorch_cos_sim(emb_5, self.embeddings_original)]
