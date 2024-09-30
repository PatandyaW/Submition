import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the data
hour_df = pd.read_csv("https://raw.githubusercontent.com/PatandyaW/Submition/main/dataset/hour.csv")
day_df = pd.read_csv("https://raw.githubusercontent.com/PatandyaW/Submition/main/dataset/day.csv")

# Data preprocessing
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

day_df['weathersit'] = day_df['weathersit'].map({1: 'Clear/Partly Cloudy', 2: 'Misty/Cloudy', 3: 'Light Snow/Rain', 4: 'Severe Weather'})
day_df['mnth'] = day_df['mnth'].map({1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'})
day_df['weekday'] = day_df['weekday'].map({0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'})
day_df['season'] = day_df['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
day_df['yr'] = day_df['yr'].map({0: 2011, 1: 2012})

# Streamlit Title
st.title("Bike Rental Analysis")

# Show dataframes
if st.checkbox("Show Hour DataFrame"):
    st.write(hour_df)

if st.checkbox("Show Day DataFrame"):
    st.write(day_df)

# Weather Situation Analysis
st.header("Bike Rentals by Weather Situation")
weather_data = day_df.groupby(by='weathersit').agg({'cnt': ['max', 'min', 'mean', 'sum']})
st.write(weather_data)

plt.figure(figsize=(10, 6))
sns.boxplot(x='weathersit', y='cnt', data=day_df)
plt.title('Bike Rentals by Weather Situation')
plt.xlabel('Weather Situation')
plt.ylabel('Total Rentals')
st.pyplot(plt)

# Monthly Analysis
st.header("Count of Bike Rentals by Month")
plt.figure(figsize=(16, 6))
sns.boxplot(x='mnth', y='cnt', data=day_df, palette=["blue", "cyan"])
plt.xlabel("Month")
plt.ylabel("Total Rides")
plt.title("Count of Bike Rentals by Month")
st.pyplot(plt)

# Daily Rentals Over Time
st.header("Number of Bikers Per Day")
plt.figure(figsize=(10, 5))
plt.plot(day_df["dteday"], day_df["cnt"], marker='o', linewidth=2)
plt.title("Number of Bikers Per Day")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
st.pyplot(plt)

# Seasonal Analysis
st.header("Bike Rentals Distribution by Season")
plt.figure(figsize=(10, 6))
sns.boxplot(x='season', y='cnt', data=day_df)
plt.title('Bike Rentals Distribution by Season')
plt.xlabel('Season')
plt.ylabel('Total Rentals')
st.pyplot(plt)

# Correlation Heatmap
st.header("Correlation Heatmap for Bike Rentals and Environmental Factors")
plt.figure(figsize=(12, 8))
correlation = hour_df[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=2)
plt.title('Correlation Heatmap')
st.pyplot(plt)

# RFM Analysis
st.header("RFM Segmentation")
# Calculate the maximum date in the hour_df for recency calculation
max_date = hour_df['dteday'].max()

# Group by registered users and calculate RFM metrics
df_rfm = hour_df.groupby('registered').agg({
    'dteday': lambda x: (max_date - x.max()).days,  # Recency
    'registered': 'count',  # Frequency
    'cnt': 'sum'  # Monetary: total bike rentals by registered users
}).rename(columns={
    'dteday': 'recency',
    'registered': 'frequency',
    'cnt': 'monetary'
})

# Calculate quantiles for scoring
quantiles = df_rfm.quantile(q=[0.25, 0.5, 0.75])

# Scoring functions
def r_score(x, p, d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
    
def fm_score(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4

# Rename the columns to match RFM terminology
df_rfm.columns = ['recency', 'frequency', 'monetary']
rfm_segmentation = df_rfm.copy()

# Apply RFM scoring
rfm_segmentation['R'] = rfm_segmentation['recency'].apply(r_score, args=('recency', quantiles))
rfm_segmentation['F'] = rfm_segmentation['frequency'].apply(fm_score, args=('frequency', quantiles))
rfm_segmentation['M'] = rfm_segmentation['monetary'].apply(fm_score, args=('monetary', quantiles))

# Combine R, F, and M scores into a single RFM Score
rfm_segmentation['RFM Score'] = rfm_segmentation['R'].map(str) + rfm_segmentation['F'].map(str) + rfm_segmentation['M'].map(str)

# Display RFM Segmentation
st.write(rfm_segmentation)

