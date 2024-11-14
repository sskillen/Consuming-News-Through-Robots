import pandas as pd

import os

# Set the directory path you want to work in
directory_path = r"C:\Robots\Consuming-News-Through-Robots"

# Change the working directory
os.chdir(directory_path)

# Verify the current working directory
print("Current working directory:", os.getcwd())
# Load Qualtrics surveys
survey1 = pd.read_csv("Main+Survey+(UPDATED)+-+NL_November+13,+2024_09.44.csv")
survey2 = pd.read_csv("Pre-Survey_November+13,+2024_09.42.csv")
survey3 = pd.read_csv("CORRECTION2.0_November+13,+2024_09.45.csv")
excel_data = pd.read_csv("Random Device list11.csv")


# remove rows that don't have answers to survey 1 (main survey) #these are participants who never did the full experiment
survey1 = survey1.dropna(subset=['check_1'])
#survey1.to_excel('droppedSurvey1.xlsx', index=False)
# Optionally, preview the first few rows to better understand the data structure
print("\nSurvey 1 Preview:\n", survey1.head())
print("\nSurvey 2 Preview:\n", survey2.head())
print("\nSurvey 3 Preview:\n", survey3.head())

# Show column names for each survey
print("Survey 1 Columns:", survey1.columns)
print("Survey 2 Columns:", survey2.columns)
print("Survey 3 Columns:", survey3.columns)

# Show column names for the Excel file
print("Excel Data Columns:", excel_data.columns)



# Preview the first few rows of the Excel data
print("\nExcel Data Preview:\n", excel_data.head())

# Specify the columns to replace
#columns_to_replace = [f"Q27_{i}" for i in range(1, 9)]

# Replace these columns in pre_survey with the same columns from correction_survey
#survey2[columns_to_replace] = survey3[columns_to_replace]

# inspect updated pre-survey with corrected statements
#survey2.to_excel('corrected.xlsx', index=False)

# List of columns to remove
columns_to_remove = [
    'StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress',
    'Duration (in seconds)', 'Finished', 'RecordedDate', 'ResponseId',
    'RecipientLastName', 'RecipientFirstName', 'RecipientEmail',
    'ExternalReference', 'LocationLatitude', 'LocationLongitude',
    'DistributionChannel', 'UserLanguage'
]

# Remove columns from each survey
survey11 = survey1.drop(columns=columns_to_remove, errors='ignore')
survey21 = survey2.drop(columns=columns_to_remove, errors='ignore')
survey31 = survey3.drop(columns=columns_to_remove, errors='ignore')



# Show column names for each survey
print("Survey 1 Columns:", survey11.columns)
print("Survey 2 Columns:", survey21.columns)
print("Survey 3 Columns:", survey31.columns)

# Show column names for the Excel file
print("Excel Data Columns:", excel_data.columns)

# Rename columns in each survey and Excel data
survey12 = survey11.rename(columns={"email": "Email"})
survey22 = survey21.rename(columns={"email address_2": "Email"})
survey32 = survey31.rename(columns={"Q1": "Email"})

# Verify changes
print("Survey 1 columns:", survey12.columns)
print("Survey 2 columns:", survey22.columns)
print("Survey 3 columns:", survey32.columns)
print("Excel data columns:", excel_data.columns)

# Strip whitespace and convert to lowercase for consistency
survey12["Email"] = survey12["Email"].str.strip().str.lower()
survey22["Email"] = survey22["Email"].str.strip().str.lower()
survey32["Email"] = survey32["Email"].str.strip().str.lower()
excel_data["Email"] = excel_data["Email"].str.strip().str.lower()

# Merge survey1 and survey2 on the "Email" column
merged_data = pd.merge(survey12, survey22, on="Email", how="outer")

# Merge the result with survey3 on the "Email" column
merged_data = pd.merge(merged_data, survey32, on="Email", how="outer")

# Merge the result with excel_data on the "Email" column
merged_data = pd.merge(merged_data, excel_data, on="Email", how="outer")

print(merged_data)


#check original # of rows
print(f"Survey 1 rows: {len(survey1)}")
print(f"Survey 2 rows: {len(survey2)}")
print(f"Survey 3 rows: {len(survey3)}")
print(f"Excel Data rows: {len(excel_data)}")

# number of rows in merged data
print(f"Merged Data rows: {len(merged_data)}")

# drop NA's from Email column
merged_data2 = merged_data.dropna(subset=["Email"])
print(merged_data2)

# remove rows that don't have answers to survey 1 (main survey) #these are participants who never did the full experiment
merged_data3 = merged_data2.dropna(subset=['check_1'])

# Assuming merge_data2 is your DataFrame
merged_data3.to_excel('merge_data3.xlsx', index=False)



# Define columns to replace
columns_to_replace = [f"Q27_{i}" for i in range(1, 9)]

# Set the email column as the index to match rows
pre_survey.set_index("email", inplace=True)
correction_survey.set_index("email", inplace=True)

# Update only matching rows
for column in columns_to_replace:
    pre_survey.loc[pre_survey.index.isin(correction_survey.index), column] = correction_survey[column]

# Reset the index to return email as a column and save the updated DataFrame
pre_survey.reset_index(inplace=True)
pre_survey.to_csv("path/to/updated_pre_survey.csv", index=False)