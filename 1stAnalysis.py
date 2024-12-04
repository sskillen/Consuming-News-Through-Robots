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

print(pre["EndDate"].head(50))  # print EndDate to better understand structure of dates. Later, dates will be used to filter out unusable data

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
print(filtered_data2)


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

# Print all column names as a list
print(list(conf_dat.columns))



# Define a mapping from text to numeric values
likert_EN_5 = {
    "Strongly disagree": 1,
    "Somewhat disagree": 2,
    "Neither agree nor disagree": 3, 
    "Somewhat agree": 4,
    "Strongly agree": 5
}

# List of columns for trust propensity and tech propensity
trust_propensity = ['PropensityTrust_1', 'PropensityTrust_2', 'PropensityTrust_3', 'PropensityTrust_4']
tech_propensity = [f'PropsensityTrustTech_{i}' for i in range(1, 7)]
print(conf_dat['PropensityTrust_1'])

# Apply the mapping to the trust and tech propensity columns
conf_dat[trust_propensity] = conf_dat[trust_propensity].applymap(lambda x: likert_EN_5.get(x, x))
conf_dat[tech_propensity] = conf_dat[tech_propensity].applymap(lambda x: likert_EN_5.get(x, x))

# Verify the results
print(conf_dat[trust_propensity].head())
print(conf_dat[tech_propensity].head())


# Reverse code the numeric values
conf_dat['PropsensityTrustTech_4_Rev'] = 6 - conf_dat['PropsensityTrustTech_4']

# Verify the mapping and reverse coding
print(conf_dat[['PropsensityTrustTech_4', 'PropsensityTrustTech_4_Rev']].head())
propensity_tech = ['PropsensityTrustTech_1', 'PropsensityTrustTech_2', 'PropsensityTrustTech_3', 'PropsensityTrustTech_4_Rev', 'PropsensityTrustTech_5', 'PropsensityTrustTech_6']

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



# Define a mapping from text responses to numeric values (1 to 5)
likert_5_point = {
    'Heel erg eens': 5,
    'Eens': 4,
    'Neutraal': 3,
    'Oneens': 2,
    'Heel erg oneens': 1
}

# Apply the mapping to all relevant columns
sus_columns = [
    'Usability/Performanc_1', 'Usability/Performanc_2', 'Usability/Performanc_3',
    'Usability/Performanc_4', 'Usability/Performanc_5', 'Usability/Performanc_6',
    'Usability/Performanc_7', 'Usability/Performanc_8', 'Usability/Performanc_9',
    'Usability/Performanc_10'
]

conf_dat[sus_columns] = conf_dat[sus_columns].applymap(lambda x: likert_5_point[x])

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

likert_7_point = {
    "Heel erg oneens": 1,
    "Oneens": 2,
    "Enigszins Oneens": 3,
    "Neutraal": 4,
    "Engiszins eens": 5,
    "Eens": 6,
    "Heel erg eens": 7 }

# Apply the mapping to the relevant columns
conf_dat[enjoyment] = conf_dat[enjoyment].applymap(lambda x: likert_7_point.get(x, x))

# Check the output to ensure the mapping worked correctly
print(conf_dat[enjoyment].head())

# Combine all column groups into a single list
all_columns = enjoyment + likeability + IQ + anthro

# Convert specified columns to numeric (assuming they already have numeric-like data)
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

generalNews_trust = [ 'News General Trust_1', 'News General Trust_2']
conf_dat[generalNews_trust] = conf_dat[generalNews_trust].applymap(lambda x: likert_EN_5[x])
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
    'Overall_Device_Trust', 'Overall_credibility', 'Avg_trustpropensity', 'Avg_TECHtrust', 'SUS_Score',
    'Avg_Enjoyment', 'Avg_Likeability', 'Avg_IQ', 'Avg_Anthropomorphism'
]

categorical_vars = [
    'Language', 'Gender', 'devices.used.News',
    'Gender of Robot', 'Condition',
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


from statsmodels.multivariate.manova import MANOVA
# Fit MANOVA model
manova = MANOVA.from_formula('Overall_credibility + Overall_Device_Trust ~ Communication_Style_Transactional * Device_Type_Speaker', data=data_combined)

# Get the results
print(manova.mv_test())


# +
# Count occurrences of each unique response in 'News_Frequency'
response_counts = conf_dat[['News Habits', 'News Interests']].value_counts()

# Display the response counts
print(response_counts)


# +
import statsmodels.api as sm

# List of dependent variables (DVs)
dependent_vars = [ 'Overall_credibility', 'Overall_Device_Trust']

# List of independent variables (IVs)
independent_vars = ['Age', 'News Habits_Numeric', 'News Interests_Numeric', 'Political Leaning', 'News General Trust_1', 'News General Trust_2', 'Trust in Sources_1', 'Trust in Sources_2', 'Trust in Sources_3', 'Trust in Sources_4', 'check_1', 'check_2', 'check_3', 'News Topics', 'prior exposure', '#_selected', 'Avg_trustpropensity', 'Avg_TECHtrust', 'SUS_Score', 'Avg_Enjoyment', 'Avg_Likeability', 'Avg_IQ', 'Avg_Anthropomorphism']


# Define mappings for different columns
mappings = {
    'News Habits': {
        "Never": 1,
        "Less often than once a month": 2,
        "Less often than once a week": 3,
        "Once a week": 4,
        "2-3 days a week": 5,
        "4-6 days a week": 6,
        "Once a day": 7,
        "Between 2 and 5 times a day": 8,
        "Between 6 and 10 times a day": 9,
        "More than 10 times a day": 10,
    },
    'News Interests': {
        "Not at all interested": 1,
        "Not very interested": 2,
        "Somewhat interested": 3,
        "Very interested": 4,
        "Extremely interested": 5,
    },
    # Add more mappings for other columns as needed
}

# Loop through each column and apply the appropriate mapping
for col, mapping in mappings.items():
    new_col_name = f"{col}_Numeric"  # Create a new column name with '_Numeric'
    conf_dat[new_col_name] = conf_dat[col].map(mapping)  # Apply the specific mapping

# Check the results
print(conf_dat[['News Habits', 'News Interests', 'News Habits_Numeric', 'News Interests_Numeric']].head())
print(conf_dat.columns)


# +
# Threshold for p-value to include variables in the model
p_value_threshold = 0.05

# Convert categorical variables to dummy variables
df_encoded = pd.get_dummies(conf_dat, columns=[
    'Language', 'Gender', 'Education', 'Gender of Robot', 'Communication_Style', 'Device_Type',
    '1st Piece', '2nd Piece', '3rd Piece', 'Novelty_1', 'Novelty_2','devices.used.News'
], drop_first=True)

# List of columns to be dummy coded
columns_to_dummy = ['Language', 'Gender', 'Education', 'Gender of Robot', 'Communication_Style', 
                    'Device_Type', '1st Piece', '2nd Piece', '3rd Piece','devices.used.News', 'Novelty']
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



