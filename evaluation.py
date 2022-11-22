from sentence_transformers import SentenceTransformer, util


class Evaluation:
    def __init__(self, original_words_path: str, generated_phrases_path: str):
        self.original_words_path = original_words_path
        self.generated_phrases_path = generated_phrases_path
        self.original_words = self.load_original_words()
        self.generated_phrases = self.load_generated_phrases()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings1 = self.model.encode(self.original_words, convert_to_tensor=True)
        self.embeddings2 = self.model.encode(self.generated_phrases, convert_to_tensor=True)
        self.cosine_scores = util.cos_sim(self.embeddings1, self.embeddings2)
        # It returns in the above example a 3x3 matrix with the respective cosine similarity scores for
        # all possible pairs between embeddings1 and embeddings2.
        self.normalized_scores = self.normalize_scores()

    def normalize_scores(self) -> list:
        """
        Normalize the cosine scores to be between 1 and 10
        :return:
        """
        normalized_scores = []
        for i, row in enumerate(self.cosine_scores):
            normalized_scores.append(row / row.max() * 10)
        return normalized_scores

    def load_original_words(self) -> list:
        with open(self.original_words_path, "r") as f:
            original_words = f.readlines()
        return original_words

    def load_generated_phrases(self) -> list:
        with open(self.generated_phrases_path, "r") as f:
            generated_phrases = f.readlines()
        return generated_phrases

    def get_cosine_scores(self) -> list:
        return self.cosine_scores

    def get_top_k(self, k: int) -> list:
        top_k = []
        for i, row in enumerate(self.cosine_scores):
            top_k.append(row.topk(k=k, largest=True, sorted=True))
        return top_k

    def get_top_k_indices(self, k: int) -> list:
        top_k_indices = []
        for i, row in enumerate(self.cosine_scores):
            top_k_indices.append(row.topk(k=k, largest=True, sorted=True).indices)
        return top_k_indices

    def get_top_k_values(self, k: int) -> list:
        top_k_values = []
        for i, row in enumerate(self.cosine_scores):
            top_k_values.append(row.topk(k=k, largest=True, sorted=True).values)
        return top_k_values

    def get_embeddings1(self) -> list:
        return self.embeddings1

    def get_embeddings2(self) -> list:
        return self.embeddings2

    def get_normalized_scores(self) -> list:
        return self.normalized_scores

    def print_to_file(self) -> None:
        """
        Print to a file the number of the phrase and the cosine score
        :return:
        """
        with open("cosine_scores.txt", "w") as f:
            for i in range(len(self.original_words)):
                f.write(f"{i}: {self.cosine_scores[i][i]}\n")

        with open("normalized_scores.txt", "w") as f:
            for i in range(len(self.original_words)):
                f.write(f"{i}: {self.normalized_scores[i][i]}\n")
