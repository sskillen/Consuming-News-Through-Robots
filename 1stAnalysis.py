import pandas as pd

import os

# Set the directory path you want to work in
directory_path = r"C:\Robots\Consuming-News-Through-Robots"

# Change the working directory
os.chdir(directory_path)

# Verify the current working directory
print("Current working directory:", os.getcwd())

# Load Qualtrics surveys
main = pd.read_csv("Main+Survey+(UPDATED)+-+NL_November+28,+2024_07.31.csv")
pre = pd.read_csv("Pre-Survey_November+28,+2024_07.29.csv")
corr = pd.read_csv("CORRECTION2.0_November+28,+2024_07.32.csv")
excel_data = pd.read_csv("Random Device list11.csv")

# Show column names for each survey
print("Survey 1 Columns:", main.columns)
print("Survey 2 Columns:", pre.columns)
print("Survey 3 Columns:", corr.columns)
print("Excel Data Columns:", excel_data.columns)

print(pre["EndDate"].head(4))  # print EndDate to better understand structure of dates. Later, dates will be used to filter out unusable data

# Step 1: Remove non-datetime rows based on a keyword
pre = pre[~pre["EndDate"].str.contains("ImportId", na=False)]
pre = pre[pre["EndDate"] != "End Date"]

# Step 2: Convert to datetime
pre["EndDate"] = pd.to_datetime(pre["EndDate"], errors='coerce')

# Step 3: Drop rows with Na
pre = pre.dropna(subset=["EndDate"])

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


# Merge pre-survey and main survey on the "Email" column
merged_data = pd.merge(pre2, main3, on="Email", how="outer")
print(merged_data.head())

# Merge the result with excel data on the "Email" column
merged_data1 = pd.merge(merged_data, excel_data, on="Email", how="outer")
print(merged_data1.head())

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

print(merged_data1.head())


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

import numpy as np

# Replace empty strings or unhelpful placeholders with NaN
filtered_data['check_1'] = filtered_data['check_1'].replace(
    ['', r'^\s*$', 'We vragen je om te denken aan het nieuws dat je zojuist hoorde. \n\nKies "goed" of "fout" voor de onderstaande stellingen. - De naam van het toetsel waarmee ik sprak was "Jip"', '{"ImportId":"QID1_10"}'], 
    np.nan,
    regex=True
)
print(filtered_data.head()) #data still here

# Drop rows where 'check_1' is NaN (these are people who never completed the full experiment)
filtered_data1 = filtered_data.dropna(subset=['check_1'])
print(filtered_data1.head())

# Reset index for cleaner output
filtered_data1.reset_index(drop=True, inplace=True)


#check that only meaningful values are kept
print(filtered_data1['check_1'].unique())


# Following steps are to only keep data that was completed on and after 10/29

print(filtered_data1['EndDate_x'].head())
print(filtered_data1['EndDate_y'].head())

# Convert columns to datetime format
filtered_data1['EndDate_x'] = pd.to_datetime(filtered_data1['EndDate_x'])
filtered_data1['EndDate_y'] = pd.to_datetime(filtered_data1['EndDate_y'])


# Count and print the number of participants
num_emails = len(filtered_data1['Email'])
print(f"Number of emails: {num_emails}")

# Set cutoff date
cutoff_date = pd.to_datetime("2024-10-29 00:00:00")

# Filter rows where either EndDate_x or EndDate_y is after the cutoff
filtered_data2 = filtered_data1[
    (filtered_data1['EndDate_x'] > cutoff_date) | 
    (filtered_data1['EndDate_y'] > cutoff_date)
]

# Print filtered data
print(filtered_data2.head())


# Filter the rows where the 'Follow up' column has the value 'Ja'
follow_up_yes = filtered_data2[filtered_data2['Follow up'] == 'Ja']

# Extract the email addresses of those participants
emails_follow_up = follow_up_yes['Email'].tolist()


# Print the list of emails
print(emails_follow_up)

# Save the list of emails to a text file
with open('emails.txt', 'w') as file:
    for email in emails_follow_up:
        file.write(email + '\n')




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

# Loop through all columns and apply str.strip() and str.lower() if the column has string data
for col in conf_dat:
    if conf_dat[col].dtype == 'object':  # Check if the column is a string/object type
        conf_dat[col] = conf_dat[col].str.strip().str.lower()

print(conf_dat.head()) #all data is here at this point



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

condition_counts = conf_dat['Condition'].value_counts()

# Print the number of participants in each condition
print(condition_counts)



# Step 1: Define all mappings
# Step 1: Define all mappings
mappings = {
    'News Habits': {
        "never": 1,
        "less often than once a month": 2,
        "less often than once a week": 3,
        "once a week": 4,
        "2-3 days a week": 5,
        "4-6 days a week": 6,
        "once a day": 7,
        "between 2 and 5 times a day": 8,
        "between 6 and 10 times a day": 9,
        "more than 10 times a day": 10,
    },
    'News Interests': {
        "not at all interested": 1,
        "not very interested": 2,
        "somewhat interested": 3,
        "very interested": 4,
        "extremely interested": 5,
    },
    # Add more mappings for other columns if necessary
}

label_to_value1 = {
    "beschrijft het erg slecht": 1,
    "beschrijft het slecht": 2,
    "beschrijft het enigszins slecht": 3,
    "neutraal": 4,
    "beschrijft het enigszins goed": 5,
    "beschrijft het goed": 6,
    "beschrijft het erg goed": 7,
}

device_trust_scale = {
    "heel erg oneens": 1,
    "oneens": 2,
    "neutraal": 3,
    "eens": 4,
    "heel erg eens": 5,
}
newspiece_scale = {
    "heel weinig vertrouwen": 1,
    "weinig vertrouwen": 2,
    "neutraal": 3,
    "vertrouwen": 4,
    "veel vertrouwen": 5,
}

likert_EN_5 = {
    "strongly disagree": 1,
    "somewhat disagree": 2,
    "neither agree nor disagree": 3, 
    "somewhat agree": 4,
    "strongly agree": 5
}

likert_5_pointNL = {
    'heel erg eens': 5,
    'eens': 4,
    'neutraal': 3,
    'oneens': 2,
    'heel erg oneens': 1
}

likert_7_pointNL = {
    "heel erg oneens": 1,
    "oneens": 2,
    "enigszins oneens": 3,
    "neutraal": 4,
    "engiszins eens": 5,
    "eens": 6,
    "heel erg eens": 7

}

# +
likert_robot = {
    "completely disagree": 1,
    "mostly disagree": 2,
    "slightly disagree": 3,
    "neither agree nor disagree": 4,
    "slightly agree": 5,
    "mostly agree": 6,
    "completely agree": 7
}



# -

# Step 2: Create a function to apply mappings
def apply_mapping(df, column, mapping):
    """Applies a given mapping to a column in the dataframe."""
    new_col_name = f"{column}_Numeric"  # Create a new column name with '_Numeric'
    df[new_col_name] = df[column].map(mapping)  # Apply the mapping
    return df

# Step 3: Apply the mappings to relevant columns
def preprocess_data(df):
    # Apply mappings for specified columns
    for col, mapping in mappings.items():
        df = apply_mapping(df, col, mapping)

    # Apply Likert scale transformations for trust columns
    trust_columns = [f'Trust in Information_{i}' for i in range(1, 4)]
    df[trust_columns] = df[trust_columns].applymap(lambda x: label_to_value1.get(x, x))

    # Apply trust piece mappings
    newspiece_trust = ['trust-VVD', 'trust-student grant', 'trust-statistic', 'trust-climate', 'trust-judges']
    df[newspiece_trust] = df[newspiece_trust].applymap(lambda x: newspiece_scale.get(x, None))

    # Apply Likert scale mappings for various columns
    trust_propensity = ['PropensityTrust_1', 'PropensityTrust_2', 'PropensityTrust_3', 'PropensityTrust_4']
    tech_propensity = [f'PropsensityTrustTech_{i}' for i in range(1, 7)]
    df[trust_propensity] = df[trust_propensity].applymap(lambda x: likert_EN_5.get(x, x))
    df[tech_propensity] = df[tech_propensity].applymap(lambda x: likert_EN_5.get(x, x))

    # Apply 5-point Likert scale for usability/performance columns
    sus_columns = [
        'Usability/Performanc_1', 'Usability/Performanc_2', 'Usability/Performanc_3',
        'Usability/Performanc_4', 'Usability/Performanc_5', 'Usability/Performanc_6',
        'Usability/Performanc_7', 'Usability/Performanc_8', 'Usability/Performanc_9',
        'Usability/Performanc_10'
    ]
    df[sus_columns] = df[sus_columns].applymap(lambda x: likert_5_pointNL.get(x, x))

    # Apply 7-point Likert scale for enjoyment columns
    enjoyment_columns = ['Enjoyment_1', 'Enjoyment_2', 'Enjoyment_3', 'Enjoyment_4', 'Enjoyment_5', 'Enjoyment_6']  # Example column names
    df[enjoyment_columns] = df[enjoyment_columns].applymap(lambda x: likert_7_pointNL.get(x, x))

    # Apply general news trust transformation
    generalNews_trust = [ 'News General Trust_1', 'News General Trust_2']  # Replace with actual columns
    df[generalNews_trust] = df[generalNews_trust].applymap(lambda x: likert_EN_5[x] if x in likert_EN_5 else x)

    # Map Likert labels to numeric values for all trust columns
    device_trust_columns = [f'device_trust_{i}' for i in range(1, 13)]
    df[device_trust_columns] = df[device_trust_columns].applymap(lambda x: device_trust_scale.get(x, x))

    personal_positive = ['AttitudeRobots1_1', 'AttitudeRobots1_2', 'AttitudeRobots1_3', 'AttitudeRobots1_4', 'AttitudeRobots1_5']
    personal_negative = ['AttitudeRobots1_6', 'AttitudeRobots1_7', 'AttitudeRobots1_8', 'AttitudeRobots2_1', 'AttitudeRobots2_2']
    societal_positive = ['AttitudeRobots2_3', 'AttitudeRobots2_4', 'AttitudeRobots2_5', 'AttitudeRobots2_6', 'AttitudeRobots2_7']
    societal_negative = ['AttitudeRobots2_8', 'Q27_1', 'Q27_2', 'Q27_3', 'Q27_4']
    all_robot_scales = personal_positive + personal_negative + societal_positive + societal_negative
    df[all_robot_scales] = df[all_robot_scales].applymap(lambda x: likert_robot.get(x, x))
    
    
    generalNews_trust = [ 'News General Trust_1', 'News General Trust_2']
    df[generalNews_trust] = df[generalNews_trust].applymap(lambda x: likert_EN_5.get(x, x))

    df[['AttitudeTechGeneral_1']] = df[['AttitudeTechGeneral_1']].applymap(lambda x: likert_EN_5.get(x, x))
    return df

# +
# Step 4: Apply the preprocessing function to the dataframe
conf_dat = preprocess_data(conf_dat)


def map_topics(topics):
    topics_list = topics.split(", ")  # Split by commas to handle multiple topics
    short_topics = [news_topics_short.get(topic.strip(), topic) for topic in topics_list]  # Map and strip any extra spaces
    return ", ".join(short_topics)  # Join the mapped topics back into a single string

# Apply the function to the 'News Topics' column
conf_dat['News Topics Short'] = conf_dat['News Topics'].apply(map_topics)


# Print the resulting dataframe
print(conf_dat.head())
print(conf_dat['News Topic Shorthand'])
# -

# Check the updated dataframe
print(conf_dat.head(10))


# +
# Reverse code items 1, 2, and 3 (using the Likert scale, reverse the values)
# List of columns to reverse
reverse_columns = ['device_trust_1', 'device_trust_2', 'device_trust_3']

# Convert columns to numeric, replacing non-numeric values with NaN
conf_dat[reverse_columns] = conf_dat[reverse_columns].apply(pd.to_numeric, errors='coerce')

# Check if there are any NaN values after conversion
if conf_dat[reverse_columns].isnull().any().any():
    print("Warning: NaN values found after conversion. Handling missing values...")

# Handle NaN values (e.g., fill with the median or drop them)
conf_dat[reverse_columns] = conf_dat[reverse_columns].fillna(conf_dat[reverse_columns].median())

# Perform the reverse scoring
reverse_scale = 5
conf_dat[reverse_columns] = reverse_scale + 1 - conf_dat[reverse_columns]

print(conf_dat[reverse_columns])
# -

# Calculate the overall trust score (average of all 12 items)
device_trust_columns = [f'device_trust_{i}' for i in range(1, 13)]
conf_dat['Overall_Device_Trust'] = conf_dat[device_trust_columns].mean(axis=1)

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

# +
# Create a copy of the DataFrame (this prevents fragmentation warning)
conf_dat = conf_dat.copy()

# Calculate the overall trust in information by averaging the three columns
trustinfo_columns = [f'Trust in Information_{i}' for i in range(1, 4)]
conf_dat['Overall_Trust_in_Info'] = conf_dat[trustinfo_columns].mean(axis=1)

# -

# Check the new column
print(conf_dat[['Trust in Information_1', 'Trust in Information_2', 'Trust in Information_3', 'Overall_Trust_in_Info']].head())

# Reverse code items 1, 2, and 3 (using the Likert scale, reverse the values)
reverse_scale = 5
reverse_columns1 = [f'credibility_{i}' for i in range(1, 9)]
# Ensure all values are numeric, and replace non-numeric values with NaN
conf_dat[reverse_columns1] = conf_dat[reverse_columns1].apply(pd.to_numeric, errors='coerce')

conf_dat[reverse_columns1] = reverse_scale + 1 - conf_dat[reverse_columns1]


# Calculate the overall credibility by averaging the three columns
conf_dat['Overall_Trust_News'] = conf_dat[reverse_columns1].mean(axis=1)

# Check the new column
print(conf_dat[['credibility_1', 'credibility_2', 'credibility_8', 'Overall_Trust_News']].head())

print(conf_dat.columns)

# Print all column names as a list
print(list(conf_dat.columns))

# Reverse code the numeric values
conf_dat['PropsensityTrustTech_4_Rev'] = 6 - conf_dat['PropsensityTrustTech_4']

# Verify the mapping and reverse coding
print(conf_dat[['PropsensityTrustTech_4', 'PropsensityTrustTech_4_Rev']].head())
propensity_tech = ['PropsensityTrustTech_1', 'PropsensityTrustTech_2', 'PropsensityTrustTech_3', 'PropsensityTrustTech_4_Rev', 'PropsensityTrustTech_5', 'PropsensityTrustTech_6']

trust_propensity = ['PropensityTrust_1', 'PropensityTrust_2', 'PropensityTrust_3', 'PropensityTrust_4']
tech_propensity = [f'PropsensityTrustTech_{i}' for i in range(1, 7)]
propensity_columns = trust_propensity + propensity_tech

# Apply conversion to numeric
conf_dat[propensity_columns] = conf_dat[propensity_columns].apply(pd.to_numeric, errors='coerce')

# Calculate average for propensity columns
conf_dat['Avg_trustpropensity'] = conf_dat[trust_propensity].mean(axis=1)
conf_dat['Avg_TECHtrust'] = conf_dat[propensity_tech].mean(axis=1)

print(conf_dat['Avg_TECHtrust'])
print(conf_dat['Avg_trustpropensity'])

# criterion items removed from attitude robots because they are not part of the scale

personal_positive = ['AttitudeRobots1_1', 'AttitudeRobots1_2', 'AttitudeRobots1_3', 'AttitudeRobots1_4', 'AttitudeRobots1_5']
personal_negative = ['AttitudeRobots1_6', 'AttitudeRobots1_7', 'AttitudeRobots1_8', 'AttitudeRobots2_1', 'AttitudeRobots2_2']
societal_positive = ['AttitudeRobots2_3', 'AttitudeRobots2_4', 'AttitudeRobots2_5', 'AttitudeRobots2_6', 'AttitudeRobots2_7']
societal_negative = ['AttitudeRobots2_8', 'Q27_1', 'Q27_2', 'Q27_3', 'Q27_4']

# +
# Calculate the mean of personal_positive, personal_negative, societal_positive, and societal_negative
personal_positive_mean = conf_dat[personal_positive].mean(axis=1)
personal_negative_mean = conf_dat[personal_negative].mean(axis=1)
societal_positive_mean = conf_dat[societal_positive].mean(axis=1)
societal_negative_mean = conf_dat[societal_negative].mean(axis=1)

# Combine all new columns at once using pd.concat
new_columns = pd.DataFrame({
    'personal_positive': personal_positive_mean,
    'personal_negative': personal_negative_mean,
    'societal_positive': societal_positive_mean,
    'societal_negative': societal_negative_mean
})

# Concatenate the new columns with the original DataFrame
conf_dat = pd.concat([conf_dat, new_columns], axis=1)

# -

# Apply conversion to numeric
conf_dat[[   'personal_positive',
    'personal_negative',
    'societal_positive',
    'societal_negative']] = conf_dat[['personal_positive',
    'personal_negative',
    'societal_positive',
    'societal_negative']].apply(pd.to_numeric, errors='coerce')
print(conf_dat['personal_negative'].head())

# Apply the mapping to all relevant columns
sus_columns = [
    'Usability/Performanc_1', 'Usability/Performanc_2', 'Usability/Performanc_3',
    'Usability/Performanc_4', 'Usability/Performanc_5', 'Usability/Performanc_6',
    'Usability/Performanc_7', 'Usability/Performanc_8', 'Usability/Performanc_9',
    'Usability/Performanc_10'
]

# Apply the SUS scoring rules
sus_odd_items = ['Usability/Performanc_1', 'Usability/Performanc_3', 'Usability/Performanc_5',
                 'Usability/Performanc_7', 'Usability/Performanc_9']
sus_even_items = ['Usability/Performanc_2', 'Usability/Performanc_4', 'Usability/Performanc_6',
                  'Usability/Performanc_8', 'Usability/Performanc_10']

# Convert odd items (scale position - 1)
conf_dat[sus_odd_items] = conf_dat[sus_odd_items].apply(lambda x: x - 1)

# Convert even items (5 - scale position)
conf_dat[sus_even_items] = conf_dat[sus_even_items].apply(lambda x: 5 - x)

# Sum all contributions for each participant and multiply by 2.5 to get the SUS score
conf_dat['SUS_Score'] = conf_dat[sus_odd_items + sus_even_items].sum(axis=1) * 2.5

# Output the final SUS scores
print(conf_dat[['SUS_Score']])

#average for each pariticipant on enjoyment can be taken - higher scores = higher enjoyment 
enjoyment = ['Enjoyment_1', 'Enjoyment_2', 'Enjoyment_3', 'Enjoyment_4', 'Enjoyment_5', 'Enjoyment_6']
likeability = ['Likeability_1', 'Likeability_2', 'Likeability_3', 'Likeability_4', 'Likeability_5']
IQ = ['PercievedIQ_1', 'PercievedIQ_2', 'PercievedIQ_3', 'PercievedIQ_4', 'PercievedIQ_5']
anthro = ['Anthropomorphism_1', 'Anthropomorphism_2', 'Anthropomorphism_3', 'Anthropomorphism_4', 'Anthropomorphism_5']

# Check the output to ensure the mapping worked correctly
print(conf_dat[enjoyment].head())

# Combine all column groups into a single list
all_columns = enjoyment + likeability + IQ + anthro

# Apply conversion to numeric
conf_dat[all_columns] = conf_dat[all_columns].apply(pd.to_numeric, errors='coerce')

# Calculate average for enjoyment
conf_dat['Avg_Enjoyment'] = conf_dat[enjoyment].mean(axis=1)

# Calculate average for likeability
conf_dat['Avg_Likeability'] = conf_dat[likeability].mean(axis=1)

# Calculate average for IQ
conf_dat['Avg_IQ'] = conf_dat[IQ].mean(axis=1)

# Calculate average for anthropomorphism
conf_dat['Avg_Anthropomorphism'] = conf_dat[anthro].mean(axis=1)

#Print a list of column names to get appropriate variables for correlation matrix
print(list(conf_dat.columns))

# For categorical variables (e.g., gender, language, political leaning)
categorical_columns = ['Language', 'Education', 'Gender', 'devices.used.News', 'News Topics', 'Condition', 'Communication_Style', 'Device_Type']
print(conf_dat['News Topics'])

# For numeric variables (e.g., Age, trust scores)
# Apply conversion to numeric

numeric_columns = ['Political Leaning','Age', 'News General Trust_1', 'News General Trust_2', 'Trust in Sources_1', 'Trust in Sources_2', 'Trust in Sources_3', 'Trust in Sources_4', 'Avg_trustpropensity', 'Avg_TECHtrust', 'SUS_Score', 'Avg_Enjoyment', 'Avg_Likeability', 'Avg_IQ', 'Avg_Anthropomorphism']
conf_dat[numeric_columns] = conf_dat[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Set display options to show more rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Disable line wrapping
pd.set_option('display.max_colwidth', None)  # Allow full column width display

# Now print the descriptive statistics
print(conf_dat[categorical_columns].describe())
print(conf_dat[numeric_columns].describe())

#count number participants
num_emails = len(conf_dat['Email'])
print(f"Number of emails: {num_emails}")

# List of categorical variables to analyze
frequency_columns = ['Gender', 'Education', 'Language']

# Function to truncate long labels for better readability
def truncate_labels(label, max_length=50):
    if len(label) > max_length:
        return label[:max_length] + "..."  # Truncate and add ellipsis
    return label

# Loop through each column and calculate frequency and percentage
for column in frequency_columns:
    print(f"\nFrequency Distribution for {column}:")
    counts = conf_dat[column].value_counts()  # Frequency count
    percentages = conf_dat[column].value_counts(normalize=True) * 100  # Percentage

    # Truncate labels for display in summary
    truncated_counts = counts.rename(index=truncate_labels)
    truncated_percentages = percentages.rename(index=truncate_labels)

    # Print frequency and percentage
    print(truncated_counts)
    print("\nPercentage Distribution:")
    print(truncated_percentages)

    # Create and print a summary table with truncated labels
    summary = pd.DataFrame({
        'Frequency': counts,
        'Percentage': percentages.round(2)  # Rounded to 2 decimal places
    }).rename(index=truncate_labels)

    print(f"\n{column} Summary Table:")
    print(summary)



# Separate numerical and categorical variables
numerical_vars = [
    'Age', 'Political Leaning', 'News General Trust_1', 'News General Trust_2',
    'Trust in Sources_1', 'Trust in Sources_2', 'Trust in Sources_3', 'Trust in Sources_4',
    'Overall_Device_Trust', 'Overall_Trust_News', 'Avg_trustpropensity', 'Avg_TECHtrust', 'SUS_Score',
    'Avg_Enjoyment', 'Avg_Likeability', 'Avg_IQ', 'Avg_Anthropomorphism','personal_positive',
    'personal_negative',
    'societal_positive',
    'societal_negative', 'News Habits_Numeric', 'News Interests_Numeric'
]

categorical_vars = [
    'Language', 'Gender', 'devices.used.News',
    'Gender of Robot',
    'Communication_Style', 'Device_Type', 'prior exposure','Novelty_1', 'Novelty_2'
]

# Encode categorical variables using one-hot encoding
encoded_categorical = pd.get_dummies(conf_dat[categorical_vars], drop_first=True)

# Combine numerical and encoded categorical variables
data_combined = pd.concat([conf_dat[numerical_vars], encoded_categorical], axis=1)
print(data_combined.columns)

# Compute correlation matrix (Pearson for numerical variables)
correlation_matrix = data_combined.corr()


# Set correlation threshold (e.g., |0.3| for stronger correlations)
threshold = 0.3

# Filter correlation matrix for absolute values greater than the threshold
strong_corr = correlation_matrix[(correlation_matrix.abs() >= threshold) & (correlation_matrix != 1)]

# Drop rows and columns where all values are NaN (no strong correlations)
strong_corr = strong_corr.dropna(how='all').dropna(axis=1, how='all')

# +
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'strong_corr' is your correlation matrix DataFrame

# Shorten the variable names by renaming columns and rows (if needed)
short_names = {
    "devices.used.News_Connected TV (a TV that connects to the internet via set top box, game console, other box such as Apple TV etc.)": "Apple_TV",
    "devices.used.News_Laptop or desktop computer (at work or home)": "Computer",
    "Trust in Sources_1": "Trust in NOS",
    "Trust in Sources_2": "Trust in de Telegraaf",
    "Trust in Sources_3": "Trust in de Volkskrant",
    "Trust in Sources_4": "Trust in RTL Nieuws"
    
    
}
strong_corr = strong_corr.rename(columns=short_names, index=short_names)

# Increase figure size to make space for labels
plt.figure(figsize=(12, 10))  # Larger figure size

# Masking the upper triangle for symmetry (optional)
mask = np.triu(np.ones_like(strong_corr, dtype=bool))

# Plotting the heatmap with larger annotation size and rotated axis labels
sns.heatmap(strong_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.1,
            mask=mask, annot_kws={"size": 12}, cbar_kws={'shrink': 0.8})

# Adjust axis labels and rotate them for better readability
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate x-axis labels
plt.yticks(rotation=0, fontsize=12)  # No rotation for y-axis labels

# Increase the font size of the title
plt.title("Strong Correlations (|r| ≥ 0.3)", fontsize=16)

# Adjust layout to prevent overlap
plt.tight_layout()


# -

print(list(data_combined.columns))


# +


# List of dependent variables (DVs)
dependent_vars = [ 'Overall_Trust_News', 'Overall_Device_Trust']

# List of independent variables (IVs)
independent_vars = ['Age', 'News Habits_Numeric', 'News Interests_Numeric', 'Political Leaning', 'News General Trust_1', 'News General Trust_2', 'Trust in Sources_1', 'Trust in Sources_2', 'Trust in Sources_3', 'Trust in Sources_4', 'check_1', 'check_2', 'check_3', 'News Topics', 'prior exposure', '#_selected', 'Avg_trustpropensity', 'Avg_TECHtrust', 'SUS_Score', 'Avg_Enjoyment', 'Avg_Likeability', 'Avg_IQ', 'Avg_Anthropomorphism']



# Check the results
print(conf_dat[['News Habits', 'News Interests', 'News Habits_Numeric', 'News Interests_Numeric']].head())
print(conf_dat.columns)
# -


from statsmodels.multivariate.manova import MANOVA
# Fit MANOVA model
manova = MANOVA.from_formula('Overall_Trust_News + Overall_Device_Trust ~ Communication_Style_Transactional * Device_Type_Speaker', data=data_combined)

# Get the results
print(manova.mv_test())


# +
# Count occurrences of each unique response in 'News_Frequency'
response_counts = conf_dat[['News Habits', 'News Interests']].value_counts()

# Display the response counts
print(response_counts)


# +
import statsmodels.api as sm

# Threshold for p-value to include variables in the model
p_value_threshold = 0.05

# Convert categorical variables to dummy variables
df_encoded = pd.get_dummies(conf_dat, columns=[
    'Language', 'Gender', 'Education', 'Gender of Robot', 'Communication_Style', 'Device_Type',
    '1st Piece', '2nd Piece', '3rd Piece', 'Novelty_1', 'Novelty_2','devices.used.News', 'prior exposure','check_1', 'check_2', 'check_3', 'News Topics'
], drop_first=True)

# List of columns to be dummy coded
columns_to_dummy = ['Language', 'Gender', 'Education', 'Gender of Robot', 'Communication_Style', 
                    'Device_Type', '1st Piece', '2nd Piece', '3rd Piece','devices.used.News', 'Novelty', 'prior exposure','check_1', 'check_2', 'check_3', 'News Topics']
# Filter only the dummy coded columns
dummy_coded_vars = [col for col in df_encoded.columns if any(col.startswith(c) for c in columns_to_dummy)]

# Combine the original independent variables with the dummy-coded variables
combined_independent_vars = independent_vars + dummy_coded_vars

print("Combined Independent Variables:")
print(combined_independent_vars)


# +
#correct non-numeric data types in columns

df_encoded[combined_independent_vars] = df_encoded[combined_independent_vars].astype('int64', errors='ignore')

print("\nUpdated Data Types of Independent Variables:")
print(df_encoded[combined_independent_vars].dtypes)

non_numeric_cols = df_encoded[combined_independent_vars].select_dtypes(include=['object', 'category']).columns
if len(non_numeric_cols) > 0:
    print("\nNon-Numeric Independent Variables:")
    print(non_numeric_cols)
else:
    print("\nAll Independent Variables are Numeric.")

# -

# Loop through each dependent variable and apply forward selection
for dv in dependent_vars:
    print(f"\nForward Selection for Dependent Variable: {dv}")
    
    y = df_encoded[dv]  # Current dependent variable
    X = df_encoded[independent_vars]  # All independent variables
    X = sm.add_constant(X)  # Add intercept
    
    # Start with an empty model (just the intercept)
    X_current = X[['const']]
    
    # Initialize the best model
    best_model = None
    
    # Remaining variables to consider
    remaining_vars = list(X.columns[1:])  # Exclude 'const'
    
    # Forward selection process
    while remaining_vars:
        p_values = []  # Store p-values for this step
        for var in remaining_vars:
            X_temp = X_current.copy()
            X_temp[var] = X[var]  # Add variable to the model
            
            # Fit the model
            model = sm.OLS(y, X_temp).fit()
            
            # Store the p-value of the added variable
            p_values.append((var, model.pvalues[var]))
        
        # Sort variables by p-value (ascending)
        p_values.sort(key=lambda x: x[1])
        
        # Add the variable with the lowest p-value if it's below the threshold
        if p_values[0][1] < p_value_threshold:
            selected_var = p_values[0][0]
            best_model = sm.OLS(y, X_current.assign(**{selected_var: X[selected_var]})).fit()
            X_current[selected_var] = X[selected_var]  # Add variable to the model
            remaining_vars.remove(selected_var)  # Remove selected variable
        else:
            break  # Stop if no variable meets the threshold
    
    # Print the summary of the best model for the current dependent variable
    if best_model:
        print(best_model.summary())
    else:
        print("No variables met the p-value threshold.")

# Check for NaN values in the entire DataFrame after dummy encoding
print("Missing Values in Encoded DataFrame:")
print(df_encoded.isnull().sum())



