import os

from PIL import Image
from clip_interrogator import Interrogator, Config
from tqdm import tqdm


class ImageInterrogator:
    def __init__(self, images_path: str, save_path: str = "interrogations.txt") -> None:
        self.images_path = images_path
        self.save_path = save_path
        print("Loading images...")
        self.images = self.load_images()
        self.interrogations = {}
        print("Loaded images - Ready to interrogate")

    def load_images(self) -> dict:
        """
        Loads images from self.images_path
        """
        images = {}
        pbar = tqdm(os.listdir(self.images_path))
        for image_path in pbar:
            pbar.set_description(f"Loading {image_path}")
            images[image_path] = Image.open(os.path.join(self.images_path, image_path)).convert("RGB")
        return images

    def interrogate(self) -> None:
        """
        Interrogates images in self.images and saves the results in self.save_path
        """
        ci = Interrogator(Config(clip_model_name="ViT-L/14"))
        pbar = tqdm(self.images.items())
        for image_name, image in pbar:
            pbar.set_description(f"Interrogating {image_name}")
            interrogation = ci.interrogate(image)  # this or interrogate_fast?
            self.interrogations[image_name] = interrogation
            with open(self.save_path, "a") as f:
                f.write(interrogation + "\n")

    def get_interrogations(self) -> dict:
        """
        Returns self.interrogations
        """
        return self.interrogations


if __name__ == "__main__":
    ii = ImageInterrogator(images_path="generated_images", save_path="generated_images.txt")
    ii.interrogate()
