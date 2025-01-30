import pandas as pd

# Input and output file paths
input_file = "file_IR3.xlsx"
output_file = "data_file.xlsx"

# Function to filter rows based on the given criteria
def filter_rows(df):
    # Identify columns with "sentiment" in the title
    sentiment_cols = [col for col in df.columns if "sentiment" in col]

    def row_filter(row):
        values = row[sentiment_cols].tolist()
        neg = values.count("NEG")
        pos = values.count("POS")
        neu = values.count("NEU")

        if neg > 5 or pos > 4 or neu > 4:
            return True
        if neg == 5 and neu == 2:
            return True
        if pos == 4 and neu == 3:
            return True
        return False

    # Apply the filter to the DataFrame
    return df[df.apply(row_filter, axis=1)]

# Function to categorize rows and add a new column
def categorize_row(row):
    if row["majority"] == "NEU":
        return "neutral"
    elif row["majority"] == "POS" and row["I/P"] == "P":
        return "pro-p"
    elif row["majority"] == "NEG" and row["I/P"] == "P":
        return "anti-p"
    elif row["majority"] == "POS" and row["I/P"] == "I":
        return "pro-i"
    elif row["majority"] == "NEG" and row["I/P"] == "I":
        return "anti-i"
    else:
        return "unknown"

# Load all sheets into a dictionary of DataFrames
sheets = pd.read_excel(input_file, sheet_name=None)

# Process each sheet
processed_sheets = {}
for sheet_name, df in sheets.items():
    # Filter rows
    filtered_df = filter_rows(df)

    # Add label column if the required columns exist
    if "majority" in filtered_df.columns and "I/P" in filtered_df.columns:
        filtered_df["label"] = filtered_df.apply(categorize_row, axis=1)

    # Add the processed DataFrame to the dictionary
    processed_sheets[sheet_name] = filtered_df

# Save the processed DataFrames to a new Excel file
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, processed_df in processed_sheets.items():
        processed_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Filtered and labeled data has been written to {output_file}")
