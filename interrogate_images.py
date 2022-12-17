import os

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from sentence_transformers import util, SentenceTransformer
from tqdm import tqdm


class ImageInterrogator:
    def __init__(self, images_path: str) -> None:
        self.images_path = images_path
        print("Loading images...")
        self.images = self.load_images()
        self.interrogations = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluation_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded images - Ready to interrogate")

    def load_images(self) -> dict:
        """
        Loads images from self.images_path
        In self.images_path there are multiple folders, each containing images of a certain concept
        Load them in the images dict, with the key being the folder name and the value being a
        list of images in that folder
        """
        return {
            folder: [
                Image.open(os.path.join(self.images_path, folder, image)).convert("RGB")
                for image in os.listdir(os.path.join(self.images_path, folder))
                if image.endswith(".png")
            ]
            for folder in tqdm(os.listdir(self.images_path))
        }

    def interrogate(self) -> None:
        """
        For each folder in self.images, interrogate the images in that folder
        And save in that folder a txt file with the interrogations
        """
        model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="large_coco",
                                                             is_eval=True, device=self.device)
        pbar = tqdm(self.images.items())
        for folder, images in pbar:
            # Interrogate folder only if it has not been interrogated yet (so there is not a interrogations.txt file)
            if not os.path.exists(os.path.join(self.images_path, folder, "interrogations.txt")):
                pbar.set_description(f"Interrogating images from {folder}")
                self.interrogations[folder] = self.interrogate_folder(images, vis_processors, model, folder)
                self.save_interrogations(folder)

    def interrogate_folder(self, images: list, vis_processors, model, folder_name: str) -> list:
        """
        Interrogates a list of images
        """
        name = folder_name.replace('_', ' ').replace('-', ',')
        name_embedding = self.evaluation_model.encode(name)
        interrogations = []
        for raw_image in images:
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
            captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=5)
            best_caption = max(captions, key=lambda x: util.pytorch_cos_sim(self.evaluation_model.encode(x),
                                                                            name_embedding).item())
            interrogations.append(best_caption)
        return interrogations

    def save_interrogations(self, folder: str) -> None:
        """
        Saves the interrogations of a folder in a txt file in the folder
        """
        with open(os.path.join(self.images_path, folder, "interrogations.txt"), "w") as f:
            for i, interrogation in enumerate(self.interrogations[folder]):
                f.write(f"{interrogation}\n")

    def get_interrogations(self) -> dict:
        """
        Returns self.interrogations
        """
        return self.interrogations
