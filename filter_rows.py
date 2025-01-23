import pandas as pd

# Load the Excel file
input_file = "file_IR3.xlsx"
output_file = "output1.xlsx"

# Load all sheets into a dictionary of DataFrames
sheets = pd.read_excel(input_file, sheet_name=None)

# Function to filter rows based on the given criteria
def filter_rows(df):
    # Identify columns with "sentiment" in the title
    sentiment_cols = [col for col in df.columns if "sentiment" in col]
    print(sentiment_cols)
    def row_filter(row):
        values = row[sentiment_cols].tolist()
        print(values)
        # cleaned_words = [word.strip() for word in values if word.strip()]

        neg=values.count("NEG")
        pos=values.count("POS")
        neu=values.count("NEU")
        # print("row")
        # print(neg, pos , nue)
        if neg>5:
            return True
        if pos>4:
            return True
        if neu>4:
            return True
        if neg==5 and neu==2:
            return True
        if pos==4 and neu==3:
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
