
import pandas as pd

# Load the Excel file
file_path = "filtered_output_copy.xlsx"  # Replace with your file path
output_path = "updated_excel_file.xlsx"  # Replace with your desired output file path

# Read all sheets into a dictionary of DataFrames
dfs = pd.read_excel(file_path, sheet_name=None)

# Define a function to create the new column based on the conditions
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
        return "unknown"  # Default case if none of the conditions are met


# Process each sheet
updated_sheets = {}
for sheet_name, df in dfs.items():
    df["label"] = df.apply(categorize_row, axis=1)  # Apply the function
    updated_sheets[sheet_name] = df  # Add the updated DataFrame to the dictionary

# Save the updated DataFrames back to a new Excel file
with pd.ExcelWriter(output_path) as writer:
    for sheet_name, df in updated_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("New column added to all sheets, and Excel file saved!")
