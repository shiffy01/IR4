import pandas as pd
import random

# Load the Excel file
file_path = 'output2.xlsx'
sheet_names = ['A-J', 'BBC', 'J-P', 'NY-T']

# Read all sheets into a single DataFrame
dfs = [pd.read_excel(file_path, sheet_name=sheet) for sheet in sheet_names]
combined_df = pd.concat(dfs)

# Shuffle the combined DataFrame
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Filter 125 rows per label
selected_rows = []
for label in combined_df['label'].unique():
    label_rows = combined_df[combined_df['label'] == label].sample(n=125, random_state=42)
    selected_rows.append(label_rows)

final_df = pd.concat(selected_rows)

# Save to a new Excel file
output_file = 'shuffled_file.xlsx'
final_df.to_excel(output_file, index=False)

print(f"File saved to {output_file}")
