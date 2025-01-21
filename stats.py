import pandas as pd

# Load the Excel file (change 'your_file.xlsx' to your actual file name)
file_path = 'filtered_output.xlsx'
excel_data = pd.ExcelFile(file_path)

# Dictionary to store statistics for each sheet
statistics = {}

# Iterate through each sheet in the Excel file
for sheet_name in excel_data.sheet_names:
    # Read the sheet into a DataFrame
    df = excel_data.parse(sheet_name)

    # Ensure the required columns are present
    if 'I/P' in df.columns and 'majority' in df.columns:
        # Group by the two columns and count occurrences
        counts = df.groupby(['I/P', 'majority']).size().reset_index(name='count')

        # Store the statistics for the current sheet
        statistics[sheet_name] = counts
    else:
        print(f"Sheet '{sheet_name}' is missing the required columns.")

# Display the results
for sheet, stats in statistics.items():
    print(f"\nStatistics for sheet: {sheet}")
    print(stats)

