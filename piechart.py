###########################  SPLITS INTO 4 TABS ACCORDING TO NEWSPAPER  ################################
import pandas as pd

# Assuming the dataset is saved in an Excel file
file_path = "with_prediction_and_confidence.xlsx"
df = pd.read_excel(file_path)  # Use read_excel for Excel files

# Group by 'Newspaper' column
grouped = df.groupby('Newspaper')

# Create a new Excel file with each newspaper in a separate tab
output_file = "split_by_newspaper.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:  # Use openpyxl instead
    for newspaper, group in grouped:
        # Sort rows by Sentence Number and Document Number
        sorted_group = group.sort_values(by=['Document Number','Sentence Number' ])
        sorted_group.to_excel(writer, sheet_name=newspaper, index=False)

print(f"Dataset split into 4 tabs based on the 'Newspaper' column and saved to {output_file}.")



######################  PLOTS THE PIECHART AND HISTORGAMS  #########################################3
import pandas as pd
import matplotlib.pyplot as plt

# Define the mapping of encoded labels to actual labels
label_mapping = {
    0: "anti-i",
    1: "pro-p",
    2: "neutral",
    3: "anti-p",
    4: "pro-i"
}



# Read the split file
file_path = "split_by_newspaper.xlsx"
df = pd.ExcelFile(file_path)

# Loop through each newspaper tab
for sheet_name in df.sheet_names:
    # Read the data for each newspaper
    newspaper_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Replace encoded labels with actual labels
    newspaper_df['Actual Label'] = newspaper_df['label encoded'].map(label_mapping)
    newspaper_df['Predicted Label Name'] = newspaper_df['Predicted Label'].map(label_mapping)

    # Generate a pie chart for predicted categories
    predicted_counts = newspaper_df['Predicted Label Name'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(predicted_counts, labels=predicted_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f"Predicted Categories for {sheet_name}")
    plt.show()

    # Generate a histogram for actual vs predicted categories
    plt.figure(figsize=(10, 6))
    for label, category_name in label_mapping.items():
        subset = newspaper_df[newspaper_df['Actual Label'] == category_name]
        plt.bar(category_name, (subset['Predicted Label Name'] == category_name).sum(), label=f"{category_name}")

    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.title(f"Actual vs Predicted for {sheet_name}")
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.show()
