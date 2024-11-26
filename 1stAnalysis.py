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


# List of columns to convert to categorical
columns_to_convert = ['Gender', 'Education', 'Language']

# Convert each column to categorical
for column in columns_to_convert:
    conf_dat[column] = conf_dat[column].astype('category')

# Check the data types to confirm the conversion
print(conf_dat.dtypes)

# Display a summary of the converted columns
print(conf_dat[columns_to_convert].describe())

# Education summary
print(conf_dat['Education'].describe())

# Gender distribution
print(conf_dat['Gender'].value_counts(normalize=True) * 100)

# Education distribution
print(conf_dat['Education'].value_counts(normalize=True) * 100)

print(conf_dat['Language'].value_counts(normalize=True) * 100)

# Define mapping logic
def extract_communication_style(condition):
    if 'transactional' in condition:
        return 'Transactional'
    elif 'social' in condition:
        return 'Social'
    return None

def extract_device_type(condition):
    if 'Robot' in condition:
        return 'Robot'
    elif 'Speaker' in condition:
        return 'Speaker'
    return None

# Apply mappings
conf_dat['Communication_Style'] = conf_dat['Condition'].apply(extract_communication_style)
conf_dat['Device_Type'] = conf_dat['Condition'].apply(extract_device_type)

# Display the DataFrame
print(conf_dat) 

condition_counts = conf_dat['Condition'].value_counts()

# Print the number of participants in each condition
print(condition_counts)

# Map labels to numeric values
label_to_value = {
    "Heel erg oneens": 1,
    "Oneens": 2,
    "Neutraal": 3,
    "Eens": 4,
    "Heel erg eens": 5,
}
# Map Likert labels to numeric values for all trust columns
trust_columns = [f'device_trust_{i}' for i in range(1, 13)]
conf_dat[trust_columns] = conf_dat[trust_columns].applymap(lambda x: label_to_value[x])

# Reverse code items 1, 2, and 3 (using the Likert scale, reverse the values)
reverse_scale = 5
reverse_columns = ['device_trust_1', 'device_trust_2', 'device_trust_3']
conf_dat[reverse_columns] = reverse_scale + 1 - conf_dat[reverse_columns]

# Calculate the overall trust score (average of all 12 items)
conf_dat['Overall_Device_Trust'] = conf_dat[trust_columns].mean(axis=1)

print(conf_dat['Overall_Device_Trust'])

import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind


# Convert categorical variables
conf_dat['Device_Type'] = conf_dat['Device_Type'].astype('category')
conf_dat['Communication_Style'] = conf_dat['Communication_Style'].astype('category')

# Summary statistics
print(conf_dat[['Overall_Device_Trust']].describe())

# Visualization
import matplotlib.pyplot as plt

#descriptive statistics
sns.boxplot(x='Device_Type', y='Overall_Device_Trust', data=conf_dat)
sns.boxplot(x='Communication_Style', y='Overall_Device_Trust', data=conf_dat)
sns.boxplot(x='Condition', y='Overall_Device_Trust', data=conf_dat)
# Show the plot
plt.show()

# Map labels to numeric values
label_to_value1 = {
    "Beschrijft het erg slecht": 1,
    "Beschrijft het slecht": 2,
    "Beschrijft het enigszins slecht": 3,
    "Neutraal": 4,
    "Beschrijft het enigszins goed": 5,
    "Beschrijft het goed": 6,
    "Beschrijft het erg goed": 7,
}
# Map Likert labels to numeric values for all trust columns
trustinfo_columns = [f'Trust in Information_{i}' for i in range(1, 4)]
conf_dat[trustinfo_columns] = conf_dat[trustinfo_columns].applymap(lambda x: label_to_value1[x])


# Calculate the overall trust in information by averaging the three columns
conf_dat['Overall_Trust_in_Info'] = conf_dat[trustinfo_columns].mean(axis=1)

# Check the new column
print(conf_dat[['Trust in Information_1', 'Trust in Information_2', 'Trust in Information_3', 'Overall_Trust_in_Info']].head())

# Reverse code items 1, 2, and 3 (using the Likert scale, reverse the values)
reverse_scale = 5
reverse_columns1 = [f'credibility_{i}' for i in range(1, 9)]
# Ensure all values are numeric, and replace non-numeric values with NaN
conf_dat[reverse_columns1] = conf_dat[reverse_columns1].apply(pd.to_numeric, errors='coerce')

conf_dat[reverse_columns1] = reverse_scale + 1 - conf_dat[reverse_columns1]


# Calculate the overall credibility by averaging the three columns
conf_dat['Overall_credibility'] = conf_dat[reverse_columns1].mean(axis=1)

# Check the new column
print(conf_dat[['credibility_1', 'credibility_2', 'credibility_8', 'Overall_credibility']].head())

newspiece_scale = {
    "Heel weinig vertrouwen": 1,
    "Weinig vertrouwen": 2,
    "Neutraal": 3,
    "Vertrouwen": 4,
    "Veel vertrouwen": 5,
}

# Map Likert labels to numeric values for all trust columns
newspiece_trust = ['trust-VVD', 'trust-student grant', 'trust-statistic', 'trust-climate', 'trust-judges']

# Map Likert scale labels to numeric values
conf_dat[newspiece_trust] = conf_dat[newspiece_trust].applymap(lambda x: newspiece_scale.get(x, None))


# Calculate average trust score for news items, ignoring NaN values
conf_dat['average_newspiece'] = conf_dat[newspiece_trust].mean(axis=1, skipna=True)

print(conf_dat['average_newspiece'])

print(conf_dat.columns)
print(conf_dat[['average_newspiece', 'Overall_credibility', 'Overall_Trust_in_Info']].head())



from sklearn.preprocessing import StandardScaler

# Select the columns to standardize
columns_to_standardize = ['average_newspiece', 'Overall_credibility', 'Overall_Trust_in_Info']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the selected columns
conf_dat[columns_to_standardize] = scaler.fit_transform(conf_dat[columns_to_standardize])


conf_dat['FinalTrust_in_News'] = conf_dat[['average_newspiece', 'Overall_credibility', 'Overall_Trust_in_Info']].mean(axis=1)

print(conf_dat['FinalTrust_in_News'].describe())

#Rescale to likert 1-7
conf_dat['FinalTrust_in_News_Rescaled'] = (
    (conf_dat['FinalTrust_in_News'] - conf_dat['FinalTrust_in_News'].min()) /
    (conf_dat['FinalTrust_in_News'].max() - conf_dat['FinalTrust_in_News'].min())
) * (7 - 1) + 1

print(conf_dat['FinalTrust_in_News_Rescaled'].describe())

#Histogram of distribution

conf_dat['FinalTrust_in_News_Rescaled'].hist(bins=7)
plt.title('Rescaled Trust in News')
plt.xlabel('FinalTrust_in_News_Rescaled')
plt.ylabel('Frequency')
plt.show()


# Print all column names as a list
print(list(conf_dat.columns))
# List of columns you want to combine into an average score
trust_propensity = ['PropensityTrust_1', 'PropensityTrust_2', 'PropensityTrust_3', 'PropensityTrust_4']
#one item needs reversed propensity_tech = ['PropsensityTrustTech_1', 'PropsensityTrustTech_2', 'PropsensityTrustTech_3', 'PropsensityTrustTech_4', 'PropsensityTrustTech_5', 'PropsensityTrustTech_6']
#reverse code and criterion items attitude_robots = ['AttitudeRobots1_1','AttitudeRobots1_2', 'AttitudeRobots1_3', 'AttitudeRobots1_4', 'AttitudeRobots1_5', 'AttitudeRobots1_6', 'AttitudeRobots1_7', 'AttitudeRobots1_8', 'AttitudeRobots2_1', 'AttitudeRobots2_2', 'AttitudeRobots2_3', 'AttitudeRobots2_4', 'AttitudeRobots2_5', 'AttitudeRobots2_6', 'AttitudeRobots2_7', 'AttitudeRobots2_8', 'Q27_1', 'Q27_2', 'Q27_3', 'Q27_4', 'Q27_5', 'Q27_6', 'Q27_7', 'Q27_8']
useability = ['Usability/Performanc_1', 'Usability/Performanc_2', 'Usability/Performanc_3', 'Usability/Performanc_4', 'Usability/Performanc_5', 'Usability/Performanc_6', 'Usability/Performanc_7', 'Usability/Performanc_8', 'Usability/Performanc_9', 'Usability/Performanc_10']
enjoyment = ['Enjoyment_1', 'Enjoyment_2', 'Enjoyment_3', 'Enjoyment_4', 'Enjoyment_5', 'Enjoyment_6']
likeability = ['Likeability_1', 'Likeability_2', 'Likeability_3', 'Likeability_4', 'Likeability_5']
IQ = ['PercievedIQ_1', 'PercievedIQ_2', 'PercievedIQ_3', 'PercievedIQ_4', 'PercievedIQ_5']
anthro = ['Anthropomorphism_1', 'Anthropomorphism_2', 'Anthropomorphism_3', 'Anthropomorphism_4', 'Anthropomorphism_5']
