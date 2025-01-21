import pandas as pd

# Load the Excel file (change 'your_file.xlsx' to your actual file name)
file_path = 'updated_excel_file.xlsx'
excel_data = pd.ExcelFile(file_path)

# Dictionary to store statistics for each sheet
statistics = {}

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
    else:
        print(f"Sheet '{sheet_name}' is missing the required 'label' column.")


# Display the results
for sheet, stats in statistics.items():
    print(f"\nStatistics for sheet: {sheet}")
    print(stats)
