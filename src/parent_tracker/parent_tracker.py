import csv
import os


class ParentTracker:
    def __init__(self, csv_file_path: str) -> None:
        self.csv_file_path = csv_file_path
        self.csv_file_dir = os.path.dirname(self.csv_file_path)
        self._initialize()

    def _initialize(self):
        if not os.path.exists(self.csv_file_dir):
            os.mkdir(self.csv_file_dir)

        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ["child_name", "parent_name", "parent_generation"])

    def write(self, child_name: str, parent_name: str, parent_generation: float):
        with open(self.csv_file_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([child_name, parent_name, parent_generation])
