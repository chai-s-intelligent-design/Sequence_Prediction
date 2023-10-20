import pandas as pd
import os
import shutil

# Load the behavior codes
codes_path = './datasets/clustered_data四类 新的.xlsx'
data_0_sheet = pd.read_excel(codes_path, sheet_name="0")
behavior_codes = data_0_sheet.iloc[0, 4:].dropna().values

# Exclude "类别" and create indicator vectors
behavior_codes_clean = behavior_codes[:-1]
indicator_vectors_clean = {}
for i, code in enumerate(behavior_codes_clean):
    vector = [0] * 51
    vector[i] = 1
    indicator_vectors_clean[code] = vector

# Load the data
data_path = './datasets/匹配结果-0.xlsx'
data = pd.read_excel(data_path)


# Function to identify and merge indicator vectors within the same time period
def process_and_merge_vectors(individual_data):
    new_periods = individual_data["开始时间"].notna()
    merged_vectors = []
    vectors_to_merge = []
    start_times = []
    for i, row in individual_data.iterrows():
        if new_periods.loc[i] and vectors_to_merge:
            merged_vector = [
                1 if any(vector[i] == 1 for vector in vectors_to_merge) else 0
                for i in range(51)
            ]
            merged_vectors.append(merged_vector)
            vectors_to_merge = []
        if pd.notna(row["开始时间"]):
            start_times.append(row["开始时间"])
        vectors_to_merge.append(indicator_vectors_clean[row["消费行为编码"].strip()])
    if vectors_to_merge:
        merged_vector = [
            1 if any(vectors[i] == 1 for vectors in vectors_to_merge) else 0
            for i in range(51)
        ]
        merged_vectors.append(merged_vector)
    merged_vectors_df = pd.DataFrame(merged_vectors,
                                     index=start_times).reset_index()
    return merged_vectors_df


# Function to process and export data with GBK encoding for a single individual
def process_and_export_individual(individual_data, output_dir, start_index):
    merged_vectors_df = process_and_merge_vectors(individual_data)
    file_path = os.path.join(output_dir, f"{start_index}.csv")
    merged_vectors_df.to_csv(file_path, index=False, encoding='utf-8')


# Create output directory
output_dir = './datasets/original/0'
os.makedirs(output_dir, exist_ok=True)

start_index = 0
map_file_path = os.path.join(output_dir, "map_file.txt")
with open(map_file_path, "w") as map_file:
    for seq_num in data["序号"].unique():
        individual_data = data[data["序号"] == seq_num]
        process_and_export_individual(individual_data, output_dir, start_index)
        map_file.write(f"{seq_num}\t{start_index}\n")
        start_index += 1
