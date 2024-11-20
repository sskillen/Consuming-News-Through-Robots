import pandas as pd

import os

# Set the directory path you want to work in
directory_path = r"C:\Robots\Consuming-News-Through-Robots"

# Change the working directory
os.chdir(directory_path)

# Verify the current working directory
print("Current working directory:", os.getcwd())

# Load Qualtrics surveys
main = pd.read_csv("Main+Survey+(UPDATED)+-+NL_November+20,+2024_08.33.csv")
pre = pd.read_csv("Pre-Survey_November+20,+2024_08.32.csv")
corr = pd.read_csv("CORRECTION2.0_November+20,+2024_08.33.csv")
excel_data = pd.read_csv("Random Device list11.csv")

# Show column names for each survey
print("Survey 1 Columns:", main.columns)
print("Survey 2 Columns:", pre.columns)
print("Survey 3 Columns:", corr.columns)
print("Excel Data Columns:", excel_data.columns)

print(pre["EndDate"].head(50))  # print EndDate to better understand structure of dates. Later, dates will be used to filter out unusable data

# Step 1: Remove non-datetime rows based on a keyword
pre = pre[~pre["EndDate"].str.contains("ImportId", na=False)]
pre = pre[pre["EndDate"] != "End Date"]

# Step 2: Convert to datetime
pre["EndDate"] = pd.to_datetime(pre["EndDate"], errors='coerce')

# Step 3: Drop rows with Na
pre = pre.dropna(subset=["EndDate"])
print(pre)


# List of columns to remove that are irrevelant for data analysis
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
print("Excel Data Columns:", excel_data.columns)

# Rename email columns in each survey (Emails will be used to merge the data of each participant)
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

# Verify changes
print("Survey 1 Columns:", main3.columns)
print("Survey 2 Columns:", pre2.columns)
print("Survey 3 Columns:", corr2.columns)
print("Excel data columns:", excel_data.columns)


# Define columns to replace (This is replacing the questions in the pre-survey that orginially contained a typo.)
columns_to_replace = [f"Q27_{i}" for i in range(1, 9)]

pre2.set_index('Email', inplace=True)  # Set 'Email' as the index for pre_survey
corr2.set_index('Email', inplace=True)  # Set 'Email' as the index for correction_survey

# Ensure both DataFrames have the same rows before applying isin(), but keep all data
common_emails = pre2.index.intersection(corr2.index)
print(common_emails)

# Update the rows with the common emails
pre2.loc[common_emails, columns_to_replace] = corr2.loc[common_emails, columns_to_replace]
print(pre2)

# Merge pre-survey and main survey on the "Email" column
merged_data = pd.merge(pre2, main3, on="Email", how="outer")
print(merged_data)

# Merge the result with excel data on the "Email" column
merged_data1 = pd.merge(merged_data, excel_data, on="Email", how="outer")
print(merged_data1)

# the following steps are to match different emails that a single participant used/misspelled across the various surveys
from fuzzywuzzy import fuzz, process

# Step 1: Get unique emails
emails = merged_data1['Email'].unique()

# Step 2: Create a mapping dictionary
email_mapping = {}
visited = set()


for email in emails:
    if email not in visited:
        # Find similar emails
        matches = process.extract(email, emails, scorer=fuzz.ratio)
        similar_emails = [match[0] for match in matches if match[1] > 85]  # Similarity threshold
        
        # Standardize to the first email in the group
        standard_email = similar_emails[0]
        
        # Add all similar emails to the mapping and mark them as visited
        for similar_email in similar_emails:
            email_mapping[similar_email] = standard_email
            visited.add(similar_email)

# Step 3: Add a new column for standardized emails
merged_data1['Standardized_Email'] = merged_data1['Email'].replace(email_mapping)

print(merged_data1)
merged_data1.to_csv('mappedemails.csv') #data still here at this point

# Following steps are to consolidate all data from one participant based off their standarized emails into one row

# Function to define aggregation rules dynamically
def custom_agg(series):
    if series.name == "Condition":  # Special case for the 'Condition' column
        return ', '.join(series.dropna().astype(str).unique())  # Combine all unique values
    elif series.dtype.kind in 'iufc':  # For numeric columns
        return series.sum()  # Use sum
    else:  # For other non-numeric columns
        return ', '.join(series.dropna().astype(str).unique())


# Apply aggregation
filtered_data = merged_data1.groupby('Standardized_Email').agg(custom_agg).reset_index()
filtered_data.to_csv('afteraggregation.csv') # data still all here at this point

import numpy as np

# Replace empty strings or unhelpful placeholders with NaN
filtered_data['check_1'] = filtered_data['check_1'].replace(
    ['', r'^\s*$', 'We vragen je om te denken aan het nieuws dat je zojuist hoorde. \n\nKies "goed" of "fout" voor de onderstaande stellingen. - De naam van het toetsel waarmee ik sprak was "Jip"', '{"ImportId":"QID1_10"}'], 
    np.nan,
    regex=True
)
print(filtered_data) #data still here

# Drop rows where 'check_1' is NaN (these are people who never completed the full experiment)
filtered_data1 = filtered_data.dropna(subset=['check_1'])
print(filtered_data1)

# Reset index for cleaner output
filtered_data1.reset_index(drop=True, inplace=True)
print(filtered_data1) #data still here

#check that only meaningful values are kept
print(filtered_data1['check_1'].unique())


#Following steps are to only keep data that was completed on and after 10/29

print(filtered_data1['EndDate_x'].head())
print(filtered_data1['EndDate_y'].head())

# Convert columns to datetime format
filtered_data1['EndDate_x'] = pd.to_datetime(filtered_data1['EndDate_x'])
filtered_data1['EndDate_y'] = pd.to_datetime(filtered_data1['EndDate_y'])

# Set cutoff date
cutoff_date = pd.to_datetime("2024-10-29 00:00:00")

# Filter rows where either EndDate_x or EndDate_y is after the cutoff
filtered_data2 = filtered_data1[
    (filtered_data1['EndDate_x'] > cutoff_date) | 
    (filtered_data1['EndDate_y'] > cutoff_date)
]

# Print filtered data
print(filtered_data2)

filtered_data2.to_csv('fixed.csv')



import random

# Generate unique random IDs based on the length of the DataFrame
num_ids = len(filtered_data2)
random_ids = random.sample(range(1, num_ids + 1), num_ids)

# Replace the "Email" column with the random IDs
filtered_data2["Email"] = random_ids
print(filtered_data2.head())

# Delete standarized email column for confidential data
conf_dat = filtered_data2.drop('Standardized_Email', axis=1)
print(conf_dat.head())

conf_dat.to_csv('conf_dat3.csv') 


# Convert to categorical type
conf_dat['Gender'] = conf_dat['Gender'].astype('category')

# Check the column data type and unique values
print(conf_dat['Gender'].dtype)  # Should show 'category'
print(conf_dat['Gender'].unique())  # Should show 'Male', 'Female', 'Non-binary', 'Unknown'

# Check the result
print(conf_dat['Gender'].describe())

# Age summary
print(conf_dat['Age'].describe())

# Gender distribution
print(filtered_data['Gender'].value_counts(normalize=True) * 100)

# Education distribution
print(filtered_data['Education'].value_counts(normalize=True) * 100)