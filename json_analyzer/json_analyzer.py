import glob
import json
import os
import sys
from collections import Counter
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk import agreement
from sklearn.metrics import cohen_kappa_score, classification_report


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def plot_seconds(data: list[dict]) -> None:
    """
    Plot the time spent on each question.
    :param data: The data to analyze.
    :return: None
    """
    new_data = np.array([d["timeDiffs"] for d in data])

    # Remove any values that fall outside the threshold
    filtered_data = reject_outliers(new_data, m=4.)

    sns.displot(data=filtered_data, kde=True, bins=120)
    # Set the title to the sns plot
    plt.title("Time spent on each question")
    plt.xlabel("Seconds")
    # Make the plot bigger
    plt.gcf().set_size_inches(20, 10)
    plt.axvline(filtered_data.mean(), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(filtered_data.mean() * 1.1, max_ylim * 0.9, 'Mean: {:.2f}'.format(filtered_data.mean()))
    plt.show()


def list_shared_items(coders: list) -> list:
    # Create a dictionary to store the answers for each item
    item_answers = {}

    # Iterate through the tuples and add the answers to the dictionary
    for coder, item, answer in coders:
        if item not in item_answers:
            item_answers[item] = []
        item_answers[item].append(answer)

    # Create a list to store the shared items
    shared_items = []

    # Iterate through the dictionary and check if all the answers are True
    for item, answers in item_answers.items():
        if all(answers):
            shared_items.append(item)

    return shared_items


def calculate_agreement(data: list[dict]) -> None:
    """
    Calculate the agreement between the annotators.
    :param data: The data to analyze.
    :return: None
    """
    # Truncate the arrays to the lowest of them if they are different in size
    length = min([len(d["answers"]) for d in data])

    # Create a list of tuples (coder, item, label)
    taskdata = []
    for i in range(length):
        for coder in data:
            taskdata.append((coder["name"], coder["dataset"][i], "0" if coder["answers"][i] == "middle" else "1"))
            # we append to taskdata tuples of the form (coder, item, label)

    # Create the AnnotationTask object
    ratingtask = agreement.AnnotationTask(data=taskdata)

    # Print the kappa, fleiss, alpha, and scotts scores
    print("kappa " + str(ratingtask.kappa()))
    print("fleiss " + str(ratingtask.multi_kappa()))
    print("alpha " + str(ratingtask.alpha()))
    print("scotts " + str(ratingtask.pi()))


def calculate_agreement_sliding_window(data: list[dict], window_size=100) -> None:
    """
    Calculate the agreement between the annotators using a sliding window.
    :param data: The data to analyze.
    :param window_size: The size of the sliding window.
    :return: None
    """
    # Truncate the arrays to the lowest of them if they are different in size
    length = min([len(d["answers"]) for d in data])

    # Create a list of tuples (coder, item, label)
    taskdata = []
    for i in range(length):
        for coder in data:
            taskdata.append((coder["name"], coder["dataset"][i], "0" if coder["answers"][i] == "middle" else "1"))
            # we append to taskdata tuples of the form (coder, item, label)

    # Create n ratingtasks, each with a window_size
    ratingtasks = []
    for i in range(0, length, window_size):
        ratingtasks.append(agreement.AnnotationTask(data=taskdata[i:i + window_size]))

    # Print the kappa, fleiss, alpha, and scotts scores
    print("kappa " + str([task.kappa() for task in ratingtasks]))


def calculate_coherence_index(data: list[dict]):
    for annotator in data:
        dataset = annotator["dataset"]
        answers = annotator["answers"][:len(dataset)]
        coherence_index = sum(
            (answers[i] == answers[j]) == (dataset[i] == dataset[j]) for i in range(len(dataset)) for j in
            range(i + 1, len(dataset)))
        total_pairs = sum(dataset[i] == dataset[j] for i in range(len(dataset)) for j in range(i + 1, len(dataset)))
        if total_pairs == 0:
            print("No pairs found")
        else:
            print(f"{annotator['name']} | Coherence Index: {coherence_index / total_pairs}")


def calculate_hard_probability(data: list[dict]) -> None:
    """
    Calculate the probability of every question being middle/advanced if it's hard.
    :param data: The data to analyze.
    :return: None
    """
    # Find the common answers for the hard questions
    hard_middle_answers = []
    hard_advanced_answers = []
    for d in data:
        for i, is_hard in enumerate(d["isHard"]):
            if is_hard:
                if d["answers"][i] == "middle":
                    hard_middle_answers.append(d["dataset"][i])
                else:
                    hard_advanced_answers.append(d["dataset"][i])

    print(
        f"Hard middle probability: {len(hard_middle_answers) / (len(hard_middle_answers) + len(hard_advanced_answers))}")
    print(
        f"Hard advanced probability: {len(hard_advanced_answers) / (len(hard_middle_answers) + len(hard_advanced_answers))}")


def common_hard_answers(data: list[dict], n=10) -> None:
    """
    Find the common answers for the hard questions.
    :param n: The number of common answers to print.
    :param data: The data to analyze.
    :return: None
    """
    # Find the common answers for the hard questions
    hard_answers = []
    for d in data:
        for i, is_hard in enumerate(d["isHard"]):
            if is_hard:
                hard_answers.append(d["answers"][i])

    # Associate each answer to the synset in the dataset
    hard_answers_synsets = []
    for i, d in enumerate(data):
        for j, is_hard in enumerate(d["isHard"]):
            if is_hard:
                hard_answers_synsets.append((d["dataset"][j], hard_answers[i]))

    counter = Counter(hard_answers_synsets)
    pprint(counter.most_common(n))


def common_middle_advanced_answers(data: list[dict], n=0, middle=True) -> None:
    """
    Return the top-n answers that every annotator thought as middle or as advanced
    :param middle: If True, return the middle answers, else return the advanced answers.
    :param data: The data to analyze.
    :param n: The number of common answers to print.
    :return: None
    """
    middle_answer_count = {}
    advanced_answer_count = {}
    number_of_annotators = n if n > 0 else len(data)
    # We print only the answers that every annotator thought as middle (or advanced)
    for d in data:
        for i, answer in enumerate(d["answers"]):
            if answer == "middle":
                if d["dataset"][i] not in middle_answer_count:
                    middle_answer_count[d["dataset"][i]] = 1
                middle_answer_count[d["dataset"][i]] += 1
            elif answer == "advanced":
                if d["dataset"][i] not in advanced_answer_count:
                    advanced_answer_count[d["dataset"][i]] = 1
                advanced_answer_count[d["dataset"][i]] += 1

    if middle:
        print("Middle answers:")
        for answer, count in middle_answer_count.items():
            if count == number_of_annotators:
                print(answer)
    else:
        print("Advanced answers:")
        for answer, count in advanced_answer_count.items():
            if count == number_of_annotators:
                print(answer)


def create_ground_truth(data: list[dict]) -> None:
    """
    Create a ground truth (super annotator) for the data.
    Use majority vote for the answers and the dataset.
    Create a file "super_annotator.txt" with each line like the following:
    <dataset> <answer>
    :param data: The data to analyze.
    :return: None
    """
    # Find the common answers for the hard questions
    dataset = []
    answers = []
    is_hard = []
    for i in range(len(data[0]["dataset"])):
        answers.append(Counter([d["answers"][i] for d in data]).most_common(1)[0][0])
        dataset.append(Counter([d["dataset"][i] for d in data]).most_common(1)[0][0])
        is_hard.append(Counter([d["isHard"][i] for d in data]).most_common(1)[0][0])

    with open("super_annotator.txt", "w") as f:
        for i in range(len(dataset)):
            f.write(f"{dataset[i]} --> {answers[i].replace('middle', 'basic')}\n")

    # Now write a super_annotator.json just like the other annotators
    with open("super_annotator.json", "w") as f:
        json.dump({
            "name": "super_annotator",
            "dataset": dataset,
            "answers": answers,
            "isHard": is_hard
        }, f)


def k_humans_vs_opt(data: list[dict]) -> None:
    with open("../synset_selector/synsets_with_glosses.txt", "r") as f:
        synsets = f.readlines()
    with open("../synset_selector/hyponyms_with_glosses.txt", "r") as f:
        hyponyms = f.readlines()
    with open("super_annotator.txt", "r") as f:
        super_annotator = f.readlines()
    # Create a dictionary of the super annotator: key = synset, value = answer
    # Every line in super annotator is like: <synset> --> <answer>
    super_annotator_dict = {}
    for line in super_annotator:
        synset, answer = line.split(" --> ")
        super_annotator_dict[synset] = answer.replace("\n", "")

    # Create a opt_dict where the key is the synset or the hyponym and the value is "basic" if it is in synsets
    # and "advanced" if it is in hyponyms
    opt_dict = {}
    for synset in synsets:
        if synset.strip() in super_annotator_dict or synset in super_annotator_dict:
            opt_dict[synset.strip()] = "basic"
    for hyponym in hyponyms:
        if hyponym.strip() in super_annotator_dict or hyponym in super_annotator_dict:
            opt_dict[hyponym.strip()] = "advanced"

    # Evaluate Cohen's kappa for the super annotator vs opt
    super_annotator_answers = []
    opt_answers = []
    for synset, answer in super_annotator_dict.items():
        super_annotator_answers.append(answer)
        opt_answers.append(opt_dict[synset])

    print(
        f"Cohen's kappa for the super annotator vs the opt: {cohen_kappa_score(super_annotator_answers, opt_answers)}")
    print(classification_report(super_annotator_answers, opt_answers))


def special_cases(synset_name: str):
    if "showstopper" in synset_name:
        synset_name = synset_name.replace("show, stopper", "show-stopper")
    if "yak, yack" in synset_name:
        synset_name = synset_name.replace("yakety, yak", "yakety-yak")
    if "give, and, take" in synset_name:
        synset_name = synset_name.replace("give, and, take", "give-and-take")
    if "moving, picture show" in synset_name:
        synset_name = synset_name.replace("moving, picture show", "moving-picture show").replace("motion, picture show",
                                                                                                 "motion-picture show")
    if "eye, beaming" in synset_name:
        synset_name = synset_name.replace("eye, beaming", "eye-beaming")
    if "sea, duty" in synset_name:
        synset_name = synset_name.replace("sea, duty", "sea-duty")
    if "early, morning" in synset_name:
        synset_name = synset_name.replace("early, morning", "early-morning")
    if "nitty, gritty" in synset_name:
        synset_name = synset_name.replace("nitty, gritty", "nitty-gritty")
    if "get, go" in synset_name:
        synset_name = synset_name.replace("get, go", "get-go")
    if "self, digestion" in synset_name:
        synset_name = synset_name.replace("self, digestion", "self-digestion")
    return synset_name


def k_humans_vs_stable_diffusion(data: list[dict]) -> None:
    def get_cos_scores(path_folder_inside, synset_name, image_dict, treshold=0.4):
        # Get the files inside
        files = os.listdir(path_folder_inside)
        for file in files:
            if file == "cosine_scores.txt":
                path_file = os.path.join(path_folder_inside, file)
                with open(path_file, "r") as f:
                    lines = f.readlines()
                max_score = max([float(line.replace("\n", "")) for line in lines[:5]])
                if max_score > treshold:
                    image_dict[synset_name] = "basic"
                else:
                    image_dict[synset_name] = "advanced"
        return image_dict

    path_stable_diffusion_folder = "/media/evilscript/DATAX/SD2.1/"
    with open("super_annotator.txt", "r") as f:
        super_annotator = f.readlines()
    super_annotator_dict = {}
    for line in super_annotator:
        synset, answer = line.split(" --> ")
        super_annotator_dict[synset] = answer.replace("\n", "")

    # Open the folder in path_stable_diffusion_folder and get the folders inside
    image_dict = {}
    folders = os.listdir(path_stable_diffusion_folder)  # output_middle and output_advanced

    path_folder1 = os.path.join(path_stable_diffusion_folder, folders[0])
    path_folder2 = os.path.join(path_stable_diffusion_folder, folders[1])
    if os.path.isdir(path_folder1) and os.path.isdir(path_folder2):
        # Get the folders inside
        folders_inside = os.listdir(path_folder1) + os.listdir(path_folder2)
        for folder_inside in folders_inside:
            synset_name = folder_inside.replace("_", " ").replace("-", ", ")
            synset_name = special_cases(synset_name)
            for key in super_annotator_dict.keys():
                part_to_consider = key.split(" | ")[0].split("):")[1]
                if synset_name == part_to_consider:
                    synset_name = key
                    break

            if synset_name.startswith("Synset"):
                path_folder_inside1 = os.path.join(path_folder1, folder_inside)
                path_folder_inside2 = os.path.join(path_folder2, folder_inside)
                if os.path.isdir(path_folder_inside1):
                    image_dict = get_cos_scores(path_folder_inside1, synset_name, image_dict, treshold=0.2788)
                elif os.path.isdir(path_folder_inside2):
                    image_dict = get_cos_scores(path_folder_inside2, synset_name, image_dict, treshold=0.2788)
                else:
                    print(f"{path_folder_inside1} or {path_folder_inside2} is not a folder")

    # Evaluate Cohen's kappa for the super annotator vs stable diffusion
    super_annotator_answers = []
    stable_diffusion_answers = []
    for synset, answer in super_annotator_dict.items():
        super_annotator_answers.append(answer)
        stable_diffusion_answers.append(image_dict[synset])

    print(f"Cohen's kappa for the super annotator vs the stable diffusion: "
          f"{cohen_kappa_score(super_annotator_answers, stable_diffusion_answers)}")
    print(classification_report(super_annotator_answers, stable_diffusion_answers))


def load_json_files() -> list[dict]:
    """
    Load all the json files in the current directory.

    Every dict is of the form:
    {"name": "name",
    "answers": ["answer1", "answer2", ...],
    "i": number,
    "dataset": ["synset1", "synset2", ...],
    "date": {'nanoseconds': number, 'seconds': number},
    "isHard": [True, False, ...],
    "timeDiffs": [number, number, ...]}
    :return:
    """
    # Check if there are arguments
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = glob.glob("*.json")

    # create an empty list to store the dictionaries
    dicts = []

    # loop through the list of json files
    for json_file in files:
        with open(json_file) as file:
            # load the contents of the json file as a dictionary
            data = json.load(file)
            # add the dictionary to the list
            dicts.append(data)

    return dicts


def main():
    data = load_json_files()

    options = {
        1: plot_seconds,
        2: calculate_agreement,
        3: calculate_agreement_sliding_window,
        4: calculate_coherence_index,
        5: common_hard_answers,
        6: lambda _: common_middle_advanced_answers(data, int(input(f"How many annotators (max is {len(data)}): ")),
                                                    input("Middle or advanced? (m/a): ") == "m"),
        7: calculate_hard_probability,
        8: create_ground_truth,
        9: k_humans_vs_opt,
        10: k_humans_vs_stable_diffusion,
        11: exit
    }

    while True:
        print("\n---------------------------------------------------")
        print("1. Plot the time spent on each question")
        print("2. Calculate the agreement between the annotators")
        print("3. Calculate the agreement between the annotators with a sliding window of 100")
        print("4. Calculate the coherence index for each annotator")
        print("5. Find the common answers for the hard questions")
        print("6. Find the answers that everybody thought as middle or advanced")
        print("7. Calculate the probability of a question being advanced/middle if it's hard")
        print("8. Create a ground truth (super annotator)")
        print("9. Calculate the agreement between the super annotator and opt")
        print("10. Calculate the agreement between the super annotator and stable diffusion")
        print("11. Exit")
        choice = input("Enter your choice: ")
        try:
            options[int(choice)](data)
        except (KeyError, ValueError):
            print("Invalid choice")


if __name__ == "__main__":
    main()
