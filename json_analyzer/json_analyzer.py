import glob
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from nltk import agreement


def plot_seconds(data: list[dict]) -> None:
    # TODO: fix this
    # Convert the seconds to a numpy array
    seconds = []
    for d in data:
        if "timeDiffs" in d:
            seconds.append(np.array(d["timeDiffs"]))
        else:
            print(f"Warning: {d['name']} has no timeDiffs")
    seconds = np.array(seconds)

    # Calculate the mean and standard deviation of the seconds
    mean = np.mean(seconds)
    std = np.std(seconds)

    # Create a range of values for the x-axis
    x = np.linspace(seconds.min(), seconds.max(), 100)

    # Calculate the y-values for the Gaussian distribution
    y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))

    # Plot the Gaussian distribution
    plt.plot(x, y)

    # Show the plot
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
