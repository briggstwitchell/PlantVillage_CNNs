import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd
import time
from matplotlib.dates import DateFormatter
from matplotlib.dates import HourLocator


def plot_performance(df):
    # Setting the positions and width for the bars
    pos = np.arange(len(df))
    bar_width = 0.2

    # Plotting the bar chart
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Bars for loss, accuracy, recall, and precision
    # ax1.bar(pos - 1.5*bar_width, df['loss']*100, bar_width, label='Loss', color='skyblue')
    ax1.bar(pos - 0.5*bar_width, df['accuracy']*100, bar_width, label='Accuracy', color='#e03f3f')
    ax1.bar(pos + 0.5*bar_width, df['recall']*100, bar_width, label='Recall', color='#f2915c')
    ax1.bar(pos + 1.5*bar_width, df['precision']*100, bar_width, label='Precision', color='#68c0de')

    # Setting the labels and titles
    # ax1.set_xlabel('Model Name',fontweight='bold')
    ax1.set_ylabel('Percentage (%)',fontweight='bold')
    ax1.set_title('Model Performance',fontweight='bold')
    ax1.set_xticks(pos)
    plt.xticks(rotation=45, ha='right')
    ax1.set_xticklabels(df['model_name'])
    ax1.legend(loc='upper left')

    # Twin axis for training time
    ax2 = ax1.twinx()

    ax2.scatter(df['model_name'], df['training_time'], label='Training Time (seconds)', color='#000000', marker='x', s=60)
    ax2.set_ylabel('Training Time (seconds)',fontweight='bold',rotation=270, labelpad=20)
    # ax2.set_yscale('log')  # Set a logarithmic scale for the right y-axis
    ax2.legend(loc='upper right')

    # Disable gridlines
    ax1.grid(False)
    ax2.grid(False)


    # Shrink current axis's height by 10% on the bottom
    box_1 = ax1.get_position()
    ax1.set_position([box_1.x0, box_1.y0 + box_1.height * 0.1,
                    box_1.width, box_1.height * 0.9])

    # Put a legend below current axis
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.38),
            fancybox=True, shadow=True, ncol=4)

    box_2 = ax2.get_position()
    ax2.set_position([box_2.x0, box_2.y0 + box_2.height * 0.1,
                    box_2.width, box_2.height * 0.9])

    # Put a legend below current axis
    ax2.legend(loc='upper center', bbox_to_anchor=(1.1, 1.2),
            fancybox=True, shadow=True, ncol=2)

    # Adjust layout to accommodate the external legend
    plt.subplots_adjust(right=0.75)
    plt.subplots_adjust(bottom=0.4)
    # plt.show()
    plt.savefig(f"{os.getcwd()}/models/performance/all_performance_plot.png")


def read_json_files(directory):
    json_data_list = []

    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):  # Check if the file is a JSON file
            file_path = os.path.join(directory, filename)

            # Read and parse the JSON file
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    json_data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error reading {filename}: {e}")

    df = pd.DataFrame(json_data_list)
    return df

# Usage
directory_path = f'{os.getcwd()}/models/performance/'
df = read_json_files(directory_path)
plot_performance(df)
