import pandas as pd

import os

# Set the directory path you want to work in
directory_path = r"C:\Robots\Consuming-News-Through-Robots"

# Change the working directory
os.chdir(directory_path)

# Verify the current working directory
print("Current working directory:", os.getcwd())
# Load Qualtrics surveys
main = pd.read_csv("Main+Survey+(UPDATED)+-+NL_November+13,+2024_09.44.csv")
pre = pd.read_csv("Pre-Survey_November+13,+2024_09.42.csv")
corr = pd.read_csv("CORRECTION2.0_November+13,+2024_09.45.csv")
excel_data = pd.read_csv("Random Device list11.csv")

# Optionally, preview the first few rows to better understand the data structure
print("\nSurvey 1 Preview:\n", main.head())
print("\nSurvey 2 Preview:\n", pre.head())
print("\nSurvey 3 Preview:\n", corr.head())

# Show column names for each survey
print("Survey 1 Columns:", main.columns)
print("Survey 2 Columns:", pre.columns)
print("Survey 3 Columns:", corr.columns)

# Show column names for the Excel file
print("Excel Data Columns:", excel_data.columns)



# Preview the first few rows of the Excel data
print("\nExcel Data Preview:\n", excel_data.head())






# List of columns to remove
columns_to_remove = [
    'StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress',
    'Duration (in seconds)', 'Finished', 'RecordedDate', 'ResponseId',
    'RecipientLastName', 'RecipientFirstName', 'RecipientEmail',
    'ExternalReference', 'LocationLatitude', 'LocationLongitude',
    'DistributionChannel', 'UserLanguage'
]

# Remove columns from each survey
main2 = main.drop(columns=columns_to_remove, errors='ignore')
pre1 = pre.drop(columns=columns_to_remove, errors='ignore')
corr1 = corr.drop(columns=columns_to_remove, errors='ignore')



# Show column names for each survey
print("Survey 1 Columns:", main2.columns)
print("Survey 2 Columns:", pre1.columns)
print("Survey 3 Columns:", corr1.columns)

# Show column names for the Excel file
print("Excel Data Columns:", excel_data.columns)

# Rename columns in each survey and Excel data
main3 = main2.rename(columns={"email": "Email"})
pre2 = pre1.rename(columns={"email address_2": "Email"})
corr2 = corr1.rename(columns={"Q1": "Email"})

pre2 = pre2.dropna(subset=['Email'])  # Drops rows with missing emails
# Clean the email column in each DataFrame by stripping whitespaces and converting to lowercase
main3["Email"] = main3["Email"].str.strip().str.lower()
pre2["Email"] = pre2["Email"].str.strip().str.lower()
corr2["Email"] = corr2["Email"].str.strip().str.lower()
excel_data["Email"] = excel_data["Email"].str.strip().str.lower()



# Verify changes
print("Survey 1 Columns:", main3.columns)
print("Survey 2 Columns:", pre2.columns)
print("Survey 3 Columns:", corr2.columns)
print("Excel data columns:", excel_data.columns)



# Define columns to replace
columns_to_replace = [f"Q27_{i}" for i in range(1, 9)]

# Assuming pre2 and corr2 are your DataFrames for the pre-survey and correction survey
pre2.set_index('Email', inplace=True)  # Set 'Email' as the index for pre_survey
corr2.set_index('Email', inplace=True)  # Set 'Email' as the index for correction_survey

# Ensure both DataFrames have the same rows before applying isin(), but keep all data
common_emails = pre2.index.intersection(corr2.index)
print(common_emails)

# Update the rows with the common emails
pre2.loc[common_emails, columns_to_replace] = corr2.loc[common_emails, columns_to_replace]

pre2.to_csv("pre_survey.csv")



# Merge survey1 and survey2 on the "Email" column
merged_data = pd.merge(pre2, main3, on="Email", how="outer")

# Merge the result with survey3 on the "Email" column
merged_data1 = pd.merge(merged_data, excel_data, on="Email", how="outer")


print(merged_data1)
merged_data1.to_csv("merged_data1.csv")

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


