import pandas as pd
import re
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Load in the datasets
df = pd.read_csv('googleplaystore - googleplaystore.csv')
reviews = pd.read_csv('googleplaystore_user_reviews - googleplaystore_user_reviews.csv')
print(df.shape)
print(reviews.shape)

# I noticed that there were some duplicate entries, and hence I decided to drop them
df = df.drop_duplicates()
reviews = reviews.drop_duplicates()
print(df.shape)
print(reviews.shape)

# Initialize RobustScaler and MinMaxScaler to be used later
robust_scaler = RobustScaler()
min_max_scaler = MinMaxScaler()


############## Part 1: Data Cleaning and Feature Engineering




# There are a lot of missing reviews in this dataset which render themselves useless.
# Therefore, this step cleans the data by removing all word reviews with NAN values
reviews.dropna(inplace=True)
print(reviews.shape)

# Data tidying: some installation data in the 'Installs' column are stored as string literals. Using regular expressions, we can convert the column to contain only floats
def convert_installs(installs_str):
    cleaned_str = re.sub(r'\D', '', installs_str)
    return float(cleaned_str)

df['Installs'] = df['Installs'].apply(convert_installs)

# Converted 'Last Updated' column from string literal to proper date-time format
df["Last Updated"] = pd.to_datetime(df["Last Updated"])


# Feature Engineering - Calculating the average polarity and subjectivity of each App under reviews
reviews['average_polarity'] = reviews.groupby('App')['Sentiment_Polarity'].transform('mean')
reviews['average_subjectivity'] = reviews.groupby('App')['Sentiment_Subjectivity'].transform('mean')
print(reviews.head())

# Merge the two datasets together to bring over the average_polarity and average_subjectivity ratings
merged_df = pd.merge(df, reviews[['App', 'average_polarity', 'average_subjectivity']], on='App', how='left')
merged_df = merged_df.drop_duplicates()
print(merged_df.head())
print(merged_df.shape)
print(merged_df.isnull().sum())


# Added a new metric: Objective_Positivity. The higher the polarity and the lower the subjectivity, the greater the objective_positivity score
merged_df['Objective_Positivity'] = (merged_df['average_polarity'] + (1 - merged_df['average_subjectivity'])) / 2
merged_df['Objective_Positivity'] = merged_df['Objective_Positivity'].fillna(0) # replaced all missing objective_positivity values with 0 as those apps do not have reviews to calculate the metric



# The criteria asks for the apps to be popular, but not too popular;
# To find apps that best fit this description, we create a popularity index that is highest if the app is closest to the median for number of installs
median_installs = merged_df['Installs'].median()
max_installs = merged_df['Installs'].max()
print(median_installs, max_installs)
# The Below code is used just to show the wide distribution of number of installations in this dataset
merged_df['Installs'].plot(kind='box')
plt.title("Distribution of Number of Installations")
plt.show()
np.log1p(merged_df['Installs']).plot(kind='box')
plt.title("Distribution of Number of Installations (with log transformation)")
plt.show()

def popularity_score(installs):
    return 1 - (abs(installs - median_installs) / max_installs)

merged_df['Popularity_Score'] = merged_df['Installs'].apply(popularity_score)
 # Converts scores to values ranging from 0 to 1 for ease of calculation further down the road

# Convert Ratings from values ranging from 0 to 5, to values ranging from 0 to 1 for ease of calculation
merged_df['Ratings_Score'] = min_max_scaler.fit_transform(merged_df[['Rating']])
merged_df['Ratings_Score'] = merged_df['Ratings_Score'].fillna(0)

# Relevancy: When the application is last updated can be an indicator as to the relevancy of the application
today = datetime.today()
merged_df['Days_Since_Last_Update'] = (today - merged_df["Last Updated"]).dt.days
merged_df['Last_Update_Scaled'] = (1 - min_max_scaler.fit_transform(merged_df[['Days_Since_Last_Update']]))
merged_df.drop(['Days_Since_Last_Update'], axis=1, inplace=True)
print(merged_df['Last_Update_Scaled'])


########## Part 2: Obtaining Findings
# Recap - New metrics added include Objective_Positivity, Popularity_Score, Ratings_Score, and Last_Update_Scaled.
# We now find which App is most worthy to be included in the article by creating a new metric called "Overall Score", which is the average of all the new metrics we created
merged_df["Overall Score"] = (merged_df['Objective_Positivity'] + merged_df['Popularity_Score'] + merged_df['Ratings_Score'] + merged_df['Last_Update_Scaled']) / 4
# I then group all the apps into four broad categories using mapping.
category_mapping = {
    'ART_AND_DESIGN': 'Lifestyle & Social',
    'AUTO_AND_VEHICLES': 'Lifestyle & Social',
    'BEAUTY': 'Lifestyle & Social',
    'DATING': 'Lifestyle & Social',
    'EVENTS': 'Lifestyle & Social',
    'FOOD_AND_DRINK': 'Lifestyle & Social',
    'HEALTH_AND_FITNESS': 'Lifestyle & Social',
    'HOUSE_AND_HOME': 'Lifestyle & Social',
    'LIFESTYLE': 'Lifestyle & Social',
    'MEDICAL': 'Lifestyle & Social',
    'PARENTING': 'Lifestyle & Social',
    'SOCIAL': 'Lifestyle & Social',
    'SHOPPING': 'Lifestyle & Social',
    'SPORTS': 'Lifestyle & Social',
    'TRAVEL_AND_LOCAL': 'Lifestyle & Social',
    'WEATHER': 'Lifestyle & Social',
    'BUSINESS': 'Productivity & Tools',
    'FINANCE': 'Productivity & Tools',
    'LIBRARIES_AND_DEMO': 'Productivity & Tools',
    'TOOLS': 'Productivity & Tools',
    'PERSONALIZATION': 'Productivity & Tools',
    'PRODUCTIVITY': 'Productivity & Tools',
    'MAPS_AND_NAVIGATION': 'Productivity & Tools',
    'ENTERTAINMENT': 'Entertainment & Media',
    'COMICS': 'Entertainment & Media',
    'COMMUNICATION': 'Entertainment & Media',
    'FAMILY': 'Entertainment & Media',
    'GAME': 'Entertainment & Media',
    'PHOTOGRAPHY': 'Entertainment & Media',
    'VIDEO_PLAYERS': 'Entertainment & Media',
    'NEWS_AND_MAGAZINES': 'Entertainment & Media',
    'BOOKS_AND_REFERENCE': 'Education & Reference',
    'EDUCATION': 'Education & Reference'
}

# Create the new column
merged_df['Broad Category'] = merged_df['Category'].map(category_mapping)

# Retrieve the top four free apps of each broad category\
free_apps = merged_df[merged_df['Type'] == "Free"]
idx = free_apps.groupby('Broad Category')['Overall Score'].idxmax()

highest_score_free_apps = free_apps.loc[idx]

pd.set_option('display.max_columns', None)
print(highest_score_free_apps[['App', "Broad Category", "Overall Score"]])

# Retrieve the top paid app
paid_apps = merged_df[merged_df['Type'] != "Free"]
idx = paid_apps['Overall Score'].idxmax()
highest_score_paid_app = paid_apps.loc[idx]
print(highest_score_paid_app[['App', "Broad Category", "Overall Score"]])


