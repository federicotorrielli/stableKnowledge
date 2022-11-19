import sys


def flush_dict_to_file(dictionary, file_path):
    with open(file_path, "w") as f:
        for key, value in dictionary.items():
            f.write(f"{key} -> {value}\n")


def generate():
    from image_generator import ImageGenerator
    from image_title_creator import ImageTitleCreator
    # First, create the titles for each image
    itc = ImageTitleCreator()
    synset_titles = itc.get_synset_titles()
    hyponym_titles = itc.get_hyponym_titles()
    # Then, generate the images for the middle concepts (synsets)
    ig = ImageGenerator(synset_titles, powerful_gpu=True, folder_name="generated_images_synsets")
    ig.generate_images(steps=30)
    # Generate the images for the advanced concepts (hyponyms)
    ig.set_prompt_list(hyponym_titles)
    ig.set_folder_name("generated_images_hyponyms")
    ig.generate_images(steps=30)
    flush_dict_to_file(itc.get_synset_hyponym(), "synset_hyponym.txt")


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
    # Take arguments from the command line
    if len(sys.argv) > 1:
        if sys.argv[1] == "generate":
            generate()
        elif sys.argv[1] == "interrogate":
            interrogate()
        else:
            print("Invalid argument. Please use 'generate' or 'interrogate' as argument.")
    else:
        print("Please provide an argument. Use 'generate' or 'interrogate' as argument.")


if __name__ == "__main__":
    pipeline()
