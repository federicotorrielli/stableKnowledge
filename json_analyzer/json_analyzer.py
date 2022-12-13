import glob
import json
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from nltk import agreement


def plot_seconds(data: list[dict]) -> None:
    new_data = np.array([d["timeDiffs"] for d in data])

    mu = np.mean(new_data)
    sigma = np.std(new_data)

    # Define the threshold for identifying outliers
    threshold = 3 * sigma

    # Remove any values that fall outside the threshold
    filtered_data = new_data[np.abs(new_data - mu) < threshold]

    plt.title("Time differences between answers")
    plt.xlabel("Time difference (seconds)")

    sns.set_style("whitegrid")
    sns.kdeplot(data=filtered_data, bw_method=0.5)
    plt.axvline(filtered_data.mean(), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(filtered_data.mean() * 1.1, max_ylim * 0.9, 'Mean: {:.2f}'.format(filtered_data.mean()))
    plt.hist(filtered_data, bins=100, density=True)
    plt.show()


def calculate_agreement(data: list[dict]) -> None:
    # Truncate the arrays to the lower of them if they are different in size
    length = min([len(d["answers"]) for d in data])

    # Create a list of tuples (coder, item, label)
    taskdata = []
    for i in range(length):
        for coder in data:
            taskdata.append((coder["name"], i, coder["answers"][i]))

    # Create the AnnotationTask object
    ratingtask = agreement.AnnotationTask(data=taskdata)

    # Print the kappa, fleiss, alpha, and scotts scores
    print("kappa " + str(ratingtask.kappa()))
    print("fleiss " + str(ratingtask.multi_kappa()))
    print("alpha " + str(ratingtask.alpha()))
    print("scotts " + str(ratingtask.pi()))


def load_json_files() -> list[dict]:
    """
    Load all the json files in the current directory.

    Every dict is of the form: {"name": "name",
    "answers": ["answer1", "answer2", ...],
    "i": number, "dataset": ["synset1", "synset2", ...],
    "date": {'nanoseconds': number, 'seconds': number},
    "isHard": [True, False, ...],
    "timeDiffs": [number, number, ...]}
    :return:
    """
    # Check if there are arguments
    if len(sys.argv) > 1:
        num_of_files = len(sys.argv) - 1
        files = sys.argv[1:]
    else:
        num_of_files = len(glob.glob("*.json"))
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
    calculate_agreement(data)
    plot_seconds(data)


if __name__ == "__main__":
    main()
