import sys


def generate():
    from image_generator import ImageGenerator
    from image_title_creator import ImageTitleCreator
    # First, create the titles for each image
    itc = ImageTitleCreator()
    synset_titles = itc.get_synset_titles()
    hyponym_titles = itc.get_hyponym_titles()
    synset_hyponym_couples = itc.get_synset_hyponym()
    # Time to generate the images for synsets
    ig = ImageGenerator(synset_titles, powerful_gpu=True, folder_name="generated_images_synsets")
    ig.generate_images(steps=30)
    # Time to generate the images for hyponyms
    ig2 = ImageGenerator(hyponym_titles, powerful_gpu=True, folder_name="generated_images_hyponyms")
    ig2.generate_images(steps=30)


def interrogate():
    from interrogate_images import ImageInterrogator
    sys.path.append('src/blip')
    sys.path.append('src/clip')
    sys.path.append('clip-interrogator')

    # Interrogate the generated images
    ii1 = ImageInterrogator(images_path="generated_images_synsets", save_path="generated_images_synsets.txt")
    ii1.interrogate()
    synsets_interrogations = ii1.get_interrogations()
    ii2 = ImageInterrogator(images_path="generated_images_hyponyms", save_path="generated_images_hyponyms.txt")
    ii2.interrogate()
    hyponyms_interrogations = ii2.get_interrogations()


def pipeline():
    if input("Do you want to generate images? (y/n) ") == "y":
        generate()
    if input("Do you want to interrogate images? (y/n) ") == "y":
        interrogate()


if __name__ == "__main__":
    pipeline()
