import matplotlib.pyplot as plt
import pandas as pd
import os

# Path to fitness_data.csv
fitness_df = pd.read_csv("/home/Valkyrie/training_metadata_cartpole_2022-12-10-04.05.20/fitness_tracker/fitness_data.csv")

# Path to parent_data.csv
parent_df = pd.read_csv("/home/Valkyrie/training_metadata_cartpole_2022-12-10-04.05.20/parent_tracker/parent_data.csv")

# Dir to save training summary figures.
save_file_dir = "/home/Valkyrie/training_metadata_cartpole_2022-12-10-04.05.20/training_summary_figures/"

if not os.path.exists(save_file_dir):
    os.mkdir(save_file_dir)

data_frame = fitness_df
xy_info = dict()

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
        label = agent_name + "_gen_" + str(generation)
        x_data.append(label)
        y_data.append(row.fitness_value)
        xy_info[label] = {'operation':op_name, 'fitness':row.fitness_value}

    figure_count += 1
    plt.figure(figure_count)
    plt.plot(x_data, y_data, "o-", label=agent_name)
    plt.legend()
    plt.xticks(rotation="vertical", visible=True)
    plt.savefig(os.path.join(save_file_dir, agent_name+"_fitness.png"), bbox_inches="tight")
    plt.close()

    plt.figure(0)
    plt.plot(x_data, y_data, "o-", label=agent_name)
    
    
plt.figure(0)
    
for row in parent_df.iloc:
    child_name = row.child_name
    parent_name = row.parent_name
    parent_generation = row.parent_generation
    
    child_x_label = child_name + "_gen_" + "0"
    parent_x_label = parent_name + "_gen_" + str(parent_generation)
    
    
    try:
        child_y_val = xy_info[child_x_label]['fitness']
        parent_y_val = xy_info[parent_x_label]['fitness']
        operation = xy_info[child_x_label]['operation']
    except KeyError:
        continue
    
    X = [parent_x_label, child_x_label]
    Y = [parent_y_val, child_y_val]
    
    plt.plot(X,Y, '--')
    plt.text(x=parent_x_label, y=parent_y_val,s=operation)
    
    
plt.figure(0)
plt.legend()
plt.xticks(rotation="vertical", visible=True)
plt.savefig(os.path.join(save_file_dir, "all_agents_fitness_updated.png"), bbox_inches="tight")
plt.close()
