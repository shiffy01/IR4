import pandas as pd

# Load the Excel file
input_file = "updated_excel_file.xlsx"
output_file = "neutrals.xlsx"

# Load all sheets into a dictionary of DataFrames
sheets = pd.read_excel(input_file, sheet_name=None)

# Function to filter rows based on the given criteria
def filter_rows(df):

    def row_filter(row):
        if "NEU" == row["majority"] and row["I/P"]=="P":
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


#positive will be defined not as favorable towards, but as optimistic
