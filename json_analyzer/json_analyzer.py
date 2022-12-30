import glob
import json
import sys
from collections import Counter
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk import agreement


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


def calculate_coherence_index(data: list[dict]) -> None:
    """
    For each annotator, calculate the coherence index
    using a Kendall's tau function.
    :param data:
    :return:
    """
    for d in data:
        # Take the dataset
        dataset = d["dataset"]
        # Take the answers of the annotator
        answers = d["answers"][:len(dataset)]
        number_of_concordant_pairs = 0
        number_of_discordant_pairs = 0
        number_of_pairs = 0
        for i in range(len(dataset)):
            for j in range(i + 1, len(dataset)):
                if dataset[i] == dataset[j]:
                    number_of_pairs += 1
                    if answers[i] == answers[j]:
                        number_of_concordant_pairs += 1
                    else:
                        number_of_discordant_pairs += 1
        if number_of_pairs > 0:
            print(
                f"{d['name']} | Coherence Index: {(number_of_concordant_pairs - number_of_discordant_pairs) / number_of_pairs}")
        else:
            print("No pairs found")


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


def main() -> None:
    """
    Main function.
    :return: None
    """
    data = load_json_files()

    while True:
        print("\n---------------------------------------------------")
        print("1. Plot the time spent on each question")
        print("2. Calculate the agreement between the annotators")
        print("3. Calculate the coherence index for each annotator")
        print("4. Find the common answers for the hard questions")
        print("5. Find the answers that everybody thought as middle or advanced")
        print("6. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            plot_seconds(data)
        elif choice == "2":
            calculate_agreement(data)
        elif choice == "3":
            calculate_coherence_index(data)
        elif choice == "4":
            common_hard_answers(data)
        elif choice == "5":
            common_middle_advanced_answers(data, int(input(f"How many annotators (max is {len(data)}): ")),
                                           input("Middle or advanced? (m/a): ") == "m")
        elif choice == "6":
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
