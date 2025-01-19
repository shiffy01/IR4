import pandas as pd

# Load the Excel file
input_file = "file_IR3.xlsx"
output_file = "filtered_output.xlsx"

# Load all sheets into a dictionary of DataFrames
sheets = pd.read_excel(input_file, sheet_name=None)

# Function to filter rows based on the given criteria
def filter_rows(df):
    # Identify columns with "sentiment" in the title
    sentiment_cols = [col for col in df.columns if "sentiment" in col]

    def row_filter(row):
        values = row[sentiment_cols].tolist()
        unique_values = set(values)

        # Check the conditions
        if len(unique_values) == 1:  # All 7 values are the same
            return True
        elif len(unique_values) == 2 and "NUE" in unique_values:
            if values.count("NUE") <= 2 and len(unique_values - {"NUE"}) == 1:
                return True
        return False

    # Apply the filter to the DataFrame
    return df[df.apply(row_filter, axis=1)]

# Filter and write each sheet to a new Excel file
filtered_sheets = {}
for sheet_name, df in sheets.items():
    filtered_sheets[sheet_name] = filter_rows(df)

# Write the filtered sheets to a new Excel file
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, filtered_df in filtered_sheets.items():
        filtered_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Filtered data has been written to {output_file}")
