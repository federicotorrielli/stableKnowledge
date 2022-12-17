import os

import matplotlib.pyplot as plt


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

        mean_basic = sum(basic_scores) / len(basic_scores)
        mean_advanced = sum(advanced_scores) / len(advanced_scores)
        fig, ax = plt.subplots()
        ax.barh(basic_folders, basic_scores, label=f"basic: {mean_basic:.3f}")
        ax.barh(advanced_folders, advanced_scores, label=f"advanced: {mean_advanced:.3f}")
        ax.set_xlabel("Cosine similarity")
        ax.legend()
        plt.show()


def main():
    final_analyzer = FinalAnalyzer(folder_path_basic="output_middle",
                                   folder_path_advanced="output_advanced")
    final_analyzer.plot_cosine_similarity()


if __name__ == "__main__":
    main()
