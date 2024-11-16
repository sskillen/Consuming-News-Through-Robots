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

print(pre["EndDate"].head(50))  # Replace 'pre_survey' with your DataFrame name

# Step 1: Remove non-datetime rows based on a keyword
pre = pre[~pre["EndDate"].str.contains("ImportId", na=False)]
pre = pre[pre["EndDate"] != "End Date"]

# Step 2: Convert to datetime
pre["EndDate"] = pd.to_datetime(pre["EndDate"], errors='coerce')

# Step 3: Drop rows with NaT
pre = pre.dropna(subset=["EndDate"])
print(pre)








# List of columns to remove
columns_to_remove = [
    'StartDate', 'Status', 'IPAddress', 'Progress',
    'Duration (in seconds)', 'Finished', 'RecordedDate', 'ResponseId',
    'RecipientLastName', 'RecipientFirstName', 'RecipientEmail',
    'ExternalReference', 'LocationLatitude', 'LocationLongitude',
    'DistributionChannel', 'UserLanguage'
]

# Remove columns from each survey
main2 = main.drop(columns=columns_to_remove, errors='ignore')
pre1 = pre.drop(columns=columns_to_remove, errors='ignore')
corr1 = corr.drop(columns=columns_to_remove, errors='ignore')

# Remove NA rows from excel file (pariticpants who cancelled or no showed)
excel_data = excel_data.dropna(subset= ['1st Piece'])




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

# Sort by 'Email' and 'Enddate' to keep the most recent date
pre2 = pre2.sort_values(by=['Email', 'EndDate'])

# Drop duplicates, keeping the last occurrence for each email
pre2 = pre2.drop_duplicates(subset='Email', keep='last')

print(pre2)


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


from fuzzywuzzy import fuzz, process

# Check for similar emails
emails = merged_data1['Email'].unique()
for email in emails:
    matches = process.extract(email, emails, scorer=fuzz.ratio)
    similar_emails = [match for match in matches if match[1] > 85]  # Threshold of 85 for similarity
    if len(similar_emails) > 1:
        print(f"Similar emails to {email}: {similar_emails}")






# Drop rows where 'check_1' is NaN
filtered_data = merged_data1[merged_data1["check_1"].notna()]
filtered_data.to_csv("filtered_data.csv")

import random

# Generate unique random IDs based on the length of the DataFrame
num_ids = len(filtered_data)
random_ids = random.sample(range(1, num_ids + 1), num_ids)

# Replace the "Email" column with the random IDs
filtered_data["Email"] = random_ids
print(filtered_data.head())

# Age summary
print(filtered_data['Age'].describe())

# Gender distribution
print(filtered_data['Gender'].value_counts(normalize=True) * 100)

# Education distribution
print(filtered_data['Education'].value_counts(normalize=True) * 100)