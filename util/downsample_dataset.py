import os
import argparse
import pandas as pd
from sklearn.utils import shuffle

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

# Setup argument parsing
parser = argparse.ArgumentParser(description='Downsample a dataset to ensure equal class distribution.')
parser.add_argument(
    '--downsample-size',
    type=int,
    default=50000,
    help='Number of samples to keep for each class (default: 50000, whole dataset 650000)'
)
args = parser.parse_args()

# Load the data (from root directory/data/data.csv)
file_path = os.path.join(parent_dir, 'data', 'data.csv')
data = pd.read_csv(file_path, encoding='ISO-8859-1')


# First column is label and the last is text
texts = data.iloc[:, -1]
labels = data.iloc[:, 0].apply(lambda x: 0 if x == 0 else 1).tolist()
data = data.sample(frac=1).reset_index(drop=True)

# Create a DataFrame with texts and labels
df = pd.DataFrame({'label': labels, 'text': texts})

# Separate the data by class
df_class_0 = df[df['label'] == 0]
df_class_1 = df[df['label'] == 1]

# Determine the number of samples to keep for each class
downsample_num = args.downsample_size


# Downsample to ensure equal class distribution
df_class_0_balanced = df_class_0.sample(n=downsample_num, random_state=42)
df_class_1_balanced = df_class_1.sample(n=downsample_num, random_state=42)

# Combine the balanced subsets
df_balanced = pd.concat([df_class_0_balanced, df_class_1_balanced])

# Shuffle the combined DataFrame
df_balanced = shuffle(df_balanced, random_state=42).reset_index(drop=True)

# Save the balanced dataset to a new CSV file
balanced_file_path = os.path.join(parent_dir, 'data', 'balanced_downsampled_data.csv')
df_balanced.to_csv(balanced_file_path, index=False, encoding='ISO-8859-1')

print(f"Balanced downsampled dataset saved to {balanced_file_path}")
