from transformers import AutoModelForCausalLM, AutoTokenizer


class TransFilter:
    """
    Given a dict of the form {synset: best_hyponym}, removes from the dict
    the synsets that are too complicated for humans to understand (keep only basic words)
    """

    def __init__(self, synsets: dict):
        self.synsets = synsets
        self.filtered_synsets = {}
        checkpoint = "facebook/opt-6.7b"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

    def batch_processing(self, batch_size=100):
        """
        Process the synsets in batches of size batch_size
        """
        synsets = list(self.synsets.keys())
        for i in range(0, len(synsets), batch_size):
            batch = synsets[i:i + batch_size]
            self.process_batch(batch)

    def process_batch(self, batch):
        """
        Process a batch of synsets
        """
        for synset in batch:
            if self.is_too_complicated(synset):
                print(f"Removing {synset.lemma_names()[0]}")
            else:
                self.filtered_synsets[synset] = self.synsets[synset]

    def is_too_complicated(self, synset) -> bool:
        """
        Returns True if the synset is too complicated for humans to understand
        """
        inputs = self.tokenizer.encode(
            f"Is this a simple english word (example: dog -> yes, discombobulated -> no, beach -> yes, acquaintance -> no)? {synset.lemma_names()[0]} ->",
            return_tensors="pt").to("cuda")
        outputs = self.model.generate(inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[-2:] == "no"

    def get_synsets(self) -> dict:
        return self.synsets

    def get_filtered_synsets(self) -> dict:
        return self.filtered_synsets
