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
    ig = ImageGenerator(synset_titles, folder_name="output_middle")
    ig.generate_images(steps=30)
    # Generate the images for the advanced concepts (hyponyms)
    ig.set_prompt_list(hyponym_titles)
    ig.set_folder_name("output_advanced")
    ig.generate_images(steps=30)


def interrogate():
    from interrogate_images import ImageInterrogator
    sys.path.append('src/blip')
    sys.path.append('src/clip')
    sys.path.append('clip-interrogator')

    # Interrogate the generated images
    ii1 = ImageInterrogator(images_path="output_middle")
    ii1.interrogate()
    ii2 = ImageInterrogator(images_path="output_advanced")
    ii2.interrogate()


def evaluate():
    from evaluation import Evaluation
    ev = Evaluation("synsets.txt", "output_middle")
    ev.print_to_file()
    ev = Evaluation("hyponyms.txt", "output_advanced")
    ev.print_to_file()


def pipeline():
    # Take arguments from the command line
    if len(sys.argv) > 1:
        if sys.argv[1] == "generate":
            generate()
        elif sys.argv[1] == "interrogate":
            interrogate()
        elif sys.argv[1] == "evaluate":
            evaluate()
        else:
            print("Invalid argument. Please use 'generate' or 'interrogate' or 'evaluate' as argument.")
    else:
        print("Please provide an argument. Use 'generate' or 'interrogate' or 'evaluate' as argument.")


if __name__ == "__main__":
    pipeline()
