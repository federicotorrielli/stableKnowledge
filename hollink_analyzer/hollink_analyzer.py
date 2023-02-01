import csv
import glob

from nltk.metrics.agreement import AnnotationTask


def csv_to_list(filename) -> list:
    """
    Given a csv file, get the column named "Label" and return a list of the
    values in that column.
    :param filename: The name of the csv file
    :return: A list of the values in the "Label" column
    """
    with open(filename, 'r') as f:
        if filename != "thebadone.csv":
            reader = csv.DictReader(f)
            return [row['Label'] for row in reader]
        else:
            lines = f.readlines()
            # For every line, add to a result list if b if ;b is found,
            # h if ;h is found, l if ;l is found and X if ;X is found
            result = []
            for line in lines:
                if ";b" in line:
                    result.append("b")
                elif ";h" in line:
                    result.append("h")
                elif ";l" in line:
                    result.append("l")
                elif ";X" in line:
                    result.append("X")
            return result


def main():
    """Run the main program."""
    # For each csv file in the folder, get the list of labels
    csvlist = [csv_to_list(filename) for filename in glob.glob('*.csv')]
    # Convert the list to a list of annotator data, where each annotator data
    # is of the form (annotator_name, label, "task")
    annotator_data = []
    for i, csv in enumerate(csvlist):
        for n, label in enumerate(csv):
            annotator_data.append((i, str(n), label))
    # pprint(annotator_data)
    # Calculate the agreement between each pair of lists
    task = AnnotationTask(data=annotator_data)
    print(task.multi_kappa(), task.alpha())


if __name__ == '__main__':
    main()
