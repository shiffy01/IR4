import pandas as pd

# Load the Excel file
file_path = 'output2.xlsx'
excel_data = pd.ExcelFile(file_path)

# Dictionary to store the total counts of each label across all sheets
total_counts = {}

# Iterate through each sheet in the Excel file
for sheet_name in excel_data.sheet_names:
    # Read the sheet into a DataFrame
    df = excel_data.parse(sheet_name)

    # Ensure the required columns are present
    if 'label' in df.columns:
        # Count the occurrences of each value in the 'label' column
        value_counts = df['label'].value_counts()

        # Print the counts for each value
        print(f"Value counts in the 'label' column for sheet '{sheet_name}':")
        for label, count in value_counts.items():
            print(f"{label}: {count}")
            
            # Add counts to the total_counts dictionary
            if label in total_counts:
                total_counts[label] += count
            else:
                total_counts[label] = count
    else:
        print(f"Sheet '{sheet_name}' is missing the required 'label' column.")

# Print the total counts across all sheets
print("\nTotal value counts across all sheets:")
for label, total in total_counts.items():
    print(f"{label}: {total}")
