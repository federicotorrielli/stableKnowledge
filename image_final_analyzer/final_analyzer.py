import os

import matplotlib.pyplot as plt
import numpy as np


class FinalAnalyzer:
    def __init__(self, folder_path_basic: str, folder_path_advanced: str):
        self.folder_path_basic = folder_path_basic
        self.folder_path_advanced = folder_path_advanced
        self.folder_names = self.prepare_folder()

    def prepare_folder(self):
        def process_folder_name(name):
            return name.replace('_', ' ').replace('-', ',')

        folder_names = {}
        for folder_path in (self.folder_path_basic, self.folder_path_advanced):
            for folder in os.listdir(folder_path):
                processed_folder_name = process_folder_name(folder)
                with open(f"{folder_path}/{folder}/cosine_scores.txt", "r") as f:
                    # Locate the line that has "Mean" in it
                    # Since the line is "Mean: 0.123", we remove the first 6 chars
                    for line in f.readlines():
                        if "Mean" in line:
                            folder_names[processed_folder_name] = float(line[6:])

        return folder_names

    def plot_cosine_similarity(self):
        basic_folders, advanced_folders, basic_scores, advanced_scores = [], [], [], []
        for folder, score in self.folder_names.items():
            if os.path.exists(
                    f"{self.folder_path_basic}/{folder.replace(' ', '_').replace(',', '-')}/cosine_scores.txt"):
                basic_folders.append(folder)
                basic_scores.append(score)
            elif os.path.exists(
                    f"{self.folder_path_advanced}/{folder.replace(' ', '_').replace(',', '-')}/cosine_scores.txt"):
                advanced_folders.append(folder)
                advanced_scores.append(score)
        # Reorder the folders based on the scores (in a gaussian distribution)
        basic_folders = [x for _, x in sorted(zip(basic_scores, basic_folders))]
        basic_scores = sorted(basic_scores)
        advanced_folders = [x for _, x in sorted(zip(advanced_scores, advanced_folders))]
        advanced_scores = sorted(advanced_scores)

        mean_basic = round(sum(basic_scores) / len(basic_scores), 4)
        mean_advanced = round(sum(advanced_scores) / len(advanced_scores), 4)
        # Create a plot with two subplots (barplot) overlayed on each other
        # The advanced must be on top of the basic
        indices = np.arange(len(basic_folders))
        indices2 = np.arange(len(advanced_folders))
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        ax.bar(x=indices, height=basic_scores, width=0.4, color='b', align='center')
        ax.bar(x=indices2, height=advanced_scores, width=0.4, color='r', align='center')
        ax.set_xticks(indices)
        plt.xlabel("Concepts")
        plt.ylabel("Cosine Similarity")
        plt.title(f"Mean Cosine Similarity: Basic = {mean_basic}, Advanced = {mean_advanced}")
        plt.legend(["Basic", "Advanced"])
        plt.show()


def main():
    final_analyzer = FinalAnalyzer(folder_path_basic="output_middle",
                                   folder_path_advanced="output_advanced")
    final_analyzer.plot_cosine_similarity()


if __name__ == "__main__":
    main()
