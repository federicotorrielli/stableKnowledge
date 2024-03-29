import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score, classification_report
from tqdm import tqdm


def get_concreteness_from_file(file_name: str):
    """
    Given a file called file_name, return a dictionary of words and their
    concreteness integer value. The file has one capital word per line, some spaces,
    and a number at the end.
    :param file_name:
    :return:
    """
    concreteness_dict = {}
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            word, concreteness = line.split()
            concreteness_dict[word.lower()] = int(concreteness)
    return concreteness_dict


def get_cossim_score_from_folders(folder_path: str):
    cossim_dict = {}
    # For every folder in the folder_path
    for folder in os.listdir(folder_path):
        for subfolder in os.listdir(os.path.join(folder_path, folder)):
            # The file is called cosine_scores.txt
            cos_file = os.path.join(folder_path, folder, subfolder, "cosine_scores.txt")
            with open(cos_file, 'r') as f:
                # Take the max float of the first 5 lines of the file
                lines = f.readlines()
                # max_score = max([float(line.replace("\n", "")) for line in lines[:5]])
                mean_score = np.mean([float(line.replace("\n", "")) for line in lines[:5]])
                # Take the first word in subfolder.split("-") that does not contain _
                word_list = [word for word in subfolder.split("-") if "_" not in word]
                if len(word_list) > 0:
                    word = word_list[0]
                    cossim_dict[word.lower()] = mean_score
    return cossim_dict


def calculate_agreement(concreteness_dict, cosine_dict, concreteness_threshold, cosine_threshold, report=False):
    ncd = {word: "concrete" if concreteness_dict[word] > concreteness_threshold else "abstract" for word in
           concreteness_dict}
    ncsd = {word: "concrete" if cosine_dict[word] > cosine_threshold else "abstract" for word in cosine_dict}
    concr_list = [ncd[word] for word in sorted(ncd)]
    cossim_list = [ncsd[word] for word in sorted(ncsd)]
    if report:
        return classification_report(concr_list, cossim_list)
    return cohen_kappa_score(concr_list, cossim_list, labels=["abstract", "concrete"])


def find_correct_threshold(concreteness_dict: dict, cosine_dict: dict):
    """
    Given a concreteness dictionary and a cosine similarity dictionary, find the
    perfect treshold of concreteness points and cosine similarity points that
    maximises the agreement on concreteness. The minimum threshold for concreteness is 100 and the max is 700
    while the minimum cosine similarity is -1 and the max is 1.
    :return:
    """
    max_agreement = 0
    best_concr_threshold = 0
    best_cos_threshold = 0
    agreements_with_thresholds = []
    for concr_threshold in tqdm(range(350, 550, 1)):
        for cos_threshold in np.arange(0, 0.7, 0.0001):
            agreement = calculate_agreement(concreteness_dict, cosine_dict, concr_threshold, cos_threshold)
            agreements_with_thresholds.append((agreement, concr_threshold, cos_threshold))
            if agreement > max_agreement:
                print(f"New max agreement: {agreement}|"
                      f" Concrete threshold: {concr_threshold}|"
                      f" Cosine threshold: {cos_threshold}")
                max_agreement = agreement
                best_concr_threshold = concr_threshold
                best_cos_threshold = cos_threshold
    return max_agreement, best_concr_threshold, best_cos_threshold, agreements_with_thresholds, calculate_agreement(
        concreteness_dict, cosine_dict, best_concr_threshold, best_cos_threshold, report=True)


def plot_agreements(agreements_with_thresholds: list[tuple]) -> None:
    """
    Plot the agreements with x-axis being the cosine similarity threshold and y-axis being the concreteness threshold.
    :param agreements_with_thresholds:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Cosine similarity threshold")
    ax.set_ylabel("Concreteness threshold")
    ax.set_title("Agreement Heatmap")
    agreements = [agreement for agreement, _, _ in agreements_with_thresholds]
    concr_thresholds = [concr_threshold for _, concr_threshold, _ in agreements_with_thresholds]
    cos_thresholds = [cos_threshold for _, _, cos_threshold in agreements_with_thresholds]
    ax.scatter(cos_thresholds, concr_thresholds, c=agreements, cmap="inferno")
    plt.show()
    # Save the plot to a file
    fig.savefig("hm.png")


def main():
    filename = "abs_concr.txt"
    folder_path = "/media/evilscript/DATAX/SD1.5/"
    concr_dict = get_concreteness_from_file(filename)
    cossim_dict = get_cossim_score_from_folders(folder_path)
    # Make concr_dict only have shared words
    concr_dict = {word: concr_dict[word] for word in concr_dict if word in cossim_dict}
    cossim_dict = {word: cossim_dict[word] for word in cossim_dict if word in concr_dict}
    pickle_path = "agreements_with_thresholds.pkl"

    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            agreements_with_thresholds = pickle.load(f)
            plot_agreements(agreements_with_thresholds)

    # Find the best threshold
    max_agreement, best_concr_threshold, best_cos_threshold, agreements_with_thresholds, report = find_correct_threshold(
        concr_dict, cossim_dict)
    print(f"Max agreement: {max_agreement}|"
          f" Concrete threshold: {best_concr_threshold}|"
          f" Cosine threshold: {best_cos_threshold}")
    print(report)
    # Save agreements_with_thresholds to a pickle
    with open(pickle_path, 'wb') as f:
        pickle.dump(agreements_with_thresholds, f)


if __name__ == '__main__':
    main()
