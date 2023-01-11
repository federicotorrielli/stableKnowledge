from sklearn.metrics import cohen_kappa_score, classification_report
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class TransFilter:

    def __init__(self, words: list):
        self.words = words
        self.abstract_vs_concrete_dict = {}
        checkpoint = "facebook/opt-6.7b"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
        self.process()

    def process(self):
        for word in tqdm(self.words):
            if not self.is_abstract(word):
                self.abstract_vs_concrete_dict[word] = "concrete"
            else:
                self.abstract_vs_concrete_dict[word] = "abstract"

    def is_abstract(self, word) -> bool:
        """
        Returns True if the word is abstract
        """
        inputs = self.tokenizer.encode(
            f"Is this a concrete english word? (example: dog -> yes, advantage -> no, beach -> yes, summer -> no, potato -> yes, youth -> no, ball -> yes) {word} ->",
            return_tensors="pt").to("cuda")
        outputs = self.model.generate(inputs)
        final_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[-2:]
        return "no" in final_output

    def get_words(self) -> list:
        return self.words

    def get_abstract_vs_concrete_dict(self) -> dict:
        return self.abstract_vs_concrete_dict


def get_concreteness_from_file(file_name: str):
    """
    Given a file called file_name, return a dictionary of words and their
    concreteness integer value. The file has one capital word per line, some spaces,
    and a number at the end.
    :param file_name:
    :return:
    """
    concreteness_dict = {}
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            word, concreteness = line.split()
            concreteness_dict[word.lower()] = int(concreteness)
    return concreteness_dict


def calculate_agreement(concreteness_dict: dict, opt_dict: dict, concreteness_threshold: int, report=False):
    ncd = {word: "concrete" if concreteness_dict[word] > concreteness_threshold else "abstract" for word in
           concreteness_dict}
    concr_list = [ncd[word] for word in sorted(ncd)]
    opt_list = [opt_dict[word] for word in sorted(opt_dict)]
    if report:
        return classification_report(concr_list, opt_list)
    return cohen_kappa_score(concr_list, opt_list, labels=["abstract", "concrete"])


def get_concreteness_from_opt(concreteness_dict: dict):
    return TransFilter([word for word in concreteness_dict.keys()]).get_abstract_vs_concrete_dict()


def calculate_threshold(concr_dict, opt_dict):
    best_agreement = 0
    best_concr_threshold = 0
    for threshold in range(200, 600, 1):
        agreement = calculate_agreement(concr_dict, opt_dict, threshold)
        if agreement > best_agreement:
            print(f"New best agreement: {agreement} with threshold {threshold}")
            best_agreement = agreement
            best_concr_threshold = threshold
    print(f"Best agreement: {best_agreement} with threshold {best_concr_threshold}")
    print(calculate_agreement(concr_dict, opt_dict, best_concr_threshold, report=True))


def main():
    filename = "abs_concr.txt"
    concr_dict = get_concreteness_from_file(filename)
    opt_dict = get_concreteness_from_opt(concr_dict)
    calculate_threshold(concr_dict, opt_dict)


if __name__ == '__main__':
    main()
