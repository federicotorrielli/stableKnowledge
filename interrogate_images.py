import os

from PIL import Image
from clip_interrogator import Interrogator, Config
from tqdm import tqdm


class ImageInterrogator:
    def __init__(self, images_path: str) -> None:
        self.images_path = images_path
        print("Loading images...")
        self.images = self.load_images()
        self.interrogations = {}
        print("Loaded images - Ready to interrogate")

    def load_images(self) -> dict:
        """
        Loads images from self.images_path
        In self.images_path there are multiple folders, each containing images of a certain concept
        Load them in the images dict, with the key being the folder name and the value being a
        list of images in that folder
        """
        images = {}
        pbar = tqdm(os.listdir(self.images_path))
        for folder in pbar:
            pbar.set_description(f"Loading images from {folder}")
            images[folder] = []
            for image in os.listdir(os.path.join(self.images_path, folder)):
                images[folder].append(Image.open(os.path.join(self.images_path, folder, image)).convert("RGB"))
        return images

    def interrogate(self) -> None:
        """
        For each folder in self.images, interrogate the images in that folder
        And save in that folder a txt file with the interrogations
        """
        ci = Interrogator(Config(clip_model_name="ViT-L/14"))
        pbar = tqdm(self.images.items())
        for folder, images in pbar:
            pbar.set_description(f"Interrogating images from {folder}")
            self.interrogations[folder] = self.interrogate_folder(ci, images)
            self.save_interrogations(folder)

    def interrogate_folder(self, ci, images: list) -> list:
        """
        Interrogates a list of images
        """
        interrogations = []
        for image in images:
            interrogations.append(ci.interrogate(image))
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
