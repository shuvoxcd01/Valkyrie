from agent.meta_agent.meta_agent import MetaAgent
import csv
import os

import pandas as pd
import matplotlib.pyplot as plt


class FitnessTracker:
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
                    ["base_agent_name", "agent_generation", "operation_name", "fitness_value"])

    def write(self, agent: MetaAgent, operation_name: str, fitness_value: float):
        base_agent_name = agent.tf_agent.name.split("_generation_")[0]
        agent_generation = agent.generation

        with open(self.csv_file_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([base_agent_name, agent_generation,
                            operation_name, fitness_value])

    def plot_fitness(self):
        data_frame = pd.read_csv(self.csv_file_path)
        short_name = {
            "None": "",
            "Gradient-based-training": "G",
            "Mutation": "M",
            "Crossover": "C"
        }

        for agent_name in data_frame['base_agent_name'].unique():
            x_data = []
            y_data = []

            figure_count = 0

            agent_specific_df = data_frame[data_frame['base_agent_name'] == agent_name]

            for row in agent_specific_df.iloc:
                generation = row.agent_generation
                op_name = row.operation_name
                if op_name == "None":
                    continue
                label = str(generation)+"_" + op_name
                x_data.append(label)
                y_data.append(row.fitness_value)

            figure_count += 1
            plt.figure(figure_count)
            plt.plot(x_data, y_data, "o-", label=agent_name)
            plt.legend()
            plt.xticks(rotation="vertical", visible=True)
            plt.savefig(os.path.join(self.csv_file_dir,
                        agent_name+"_fitness.png"), bbox_inches="tight")
            plt.close()

            plt.figure(0)
            plt.plot(x_data, y_data, "o-", label=agent_name)

        plt.figure(0)
        plt.legend()
        plt.xticks(rotation="vertical", visible=True)
        plt.savefig(os.path.join(self.csv_file_dir,
                    "all_agents_fitness.png"), bbox_inches="tight")
        plt.close()
