

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Ensuring reproducibility
np.random.seed(42)


# Data Generation
students = range(1, 301)
degrees = np.random.choice(['Business', 'Law', 'Engineering', 'Design'], 300)
genders = np.random.choice(['Male', 'Female'], 300)
lunch_spots = np.random.choice(['Honest Greens', 'Makkila', 'Pancake House', 'Five Guys', 'Starbucks', 'IE Cafeteria', 'New Spot'], 300)
time_spent = np.random.normal(loc=35, scale=15, size=300).astype(int)  # Average time spent in minutes
ratings = np.random.randint(1, 6, size=300)  # Ratings from 1 to 5
price_ranges = np.random.choice(['Low', 'Medium', 'High'], 300, p=[0.2, 0.6, 0.2])
sortiment = np.random.choice(['Limited', 'Diverse', 'Very Diverse'], 300, p=[0.3, 0.5, 0.2])
freshness = np.random.choice(['Average', 'Fresh', 'Very Fresh'], 300, p=[0.2, 0.5, 0.3])
instagramable = np.random.choice([True, False], 300, p=[0.6, 0.4])
atmosphere = np.random.choice(['Cozy', 'Modern', 'Elegant', 'Casual'], 300)
waiting_time = np.random.normal(loc=10, scale=5, size=300).astype(int)  # Average waiting time in minutes


# DataFrame creation
df = pd.DataFrame({
   'Student_ID': students,
   'Degree': degrees,
   'Gender': genders,
   'Lunch_Spot': lunch_spots,
   'Time_Spent': time_spent,
   'Rating': ratings,
   'Price_Range': price_ranges,
   'Sortiment': sortiment,
   'Freshness': freshness,
   'Instagramable': instagramable,
   'Atmosphere': atmosphere,
   'Waiting_Time': waiting_time
})


# Advanced Visualization: Heatmap for Correlation Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Data Preprocessing for Clustering
features = df[['Time_Spent', 'Rating', 'Waiting_Time']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# Performing K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)
# Assuming the other parts of your DataFrame df are correctly defined
# Adding the 'Dietary_Preferences' column with a mock distribution
df['Dietary_Preferences'] = np.random.choice(['No Preference', 'Vegetarian', 'Vegan', 'Gluten-Free'], size=300, p=[0.5, 0.2, 0.2, 0.1])


# Calculate the number of students by dietary preference for each lunch spot
dietary_lunch_spot_counts = df.groupby(['Lunch_Spot', 'Dietary_Preferences']).size().unstack(fill_value=0)


# Plot a stacked bar chart


# Group by 'Lunch_Spot' and 'Dietary_Preferences', then count the number of students in each group
dietary_lunch_spot_counts = df.groupby(['Lunch_Spot', 'Dietary_Preferences']).size().unstack(fill_value=0)


# Plotting the stacked bar chart
dietary_lunch_spot_counts.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
plt.title('Popularity of Lunch Spots by Dietary Preferences')
plt.xlabel('Lunch Spot')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.legend(title='Dietary Preferences', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()




# Cross-Analysis: Price Range Preference by Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Price_Range', hue='Gender', data=df, palette='Pastel2')
plt.title('Price Range Preference by Gender')
plt.show()


# Insights & Recommendations based on the analysis (not executable, conceptual)
# Generate a mock 'Dietary_Preferences' column if it doesn't exist
# This is just for demonstration; ensure your actual data includes this information
if 'Dietary_Preferences' not in df.columns:
   df['Dietary_Preferences'] = np.random.choice(['No Preference', 'Vegetarian', 'Vegan', 'Gluten-Free'], size=df.shape[0], p=[0.5, 0.2, 0.2, 0.1])


# Calculate the average rating for each lunch spot by dietary preference
average_ratings = df.groupby(['Lunch_Spot', 'Dietary_Preferences'])['Rating'].mean().reset_index()


# Plotting the bar chart
plt.figure(figsize=(14, 10))
sns.barplot(x='Lunch_Spot', y='Rating', hue='Dietary_Preferences', data=average_ratings, palette='Pastel2')


plt.title('Average Ratings by Lunch Spot and Dietary Preferences', fontsize=16)
plt.xlabel('Lunch Spot', fontsize=14)
plt.ylabel('Average Rating', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Dietary Preferences', bbox_to_anchor=(1.05, 1), loc='upper left')


plt.tight_layout()
plt.show()


# What food is missing next to the university ?


#Step 1: Survey
# Simulate survey data


import pandas as pd
import numpy as np


# Simulate survey data
np.random.seed(42)  # For reproducibility


num_students = 300


# Simulated preferences
preferences = {
   'Student_ID': np.arange(1, num_students + 1),
   'Prefers_Local_Brands': np.random.choice([True, False], num_students, p=[0.7, 0.3]),
   'Seeks_Organic_Ingredients': np.random.choice([True, False], num_students, p=[0.8, 0.2]),
   'Wants_Delivery_Service': np.random.choice([True, False], num_students, p=[0.9, 0.1]),
   'Needs_Dietary_Options': np.random.choice(['None', 'Vegan', 'Gluten-Free', 'Dairy-Free'], num_students, p=[0.2, 0.3, 0.25, 0.25]),
   'Values_Discounts_And_Rewards': np.random.choice([True, False], num_students, p=[0.85, 0.15]),
}


survey_df = pd.DataFrame(preferences)
#Step 2: Analyze Survey Data
# Analyzing preferences


preference_summary = {
   'Local Brands': survey_df['Prefers_Local_Brands'].mean(),
   'Organic Ingredients': survey_df['Seeks_Organic_Ingredients'].mean(),
   'Delivery Service': survey_df['Wants_Delivery_Service'].mean(),
   'Discounts and Rewards': survey_df['Values_Discounts_And_Rewards'].mean(),
}


# Count dietary needs
dietary_needs_counts = survey_df['Needs_Dietary_Options'].value_counts()


# Displaying preferences summary
for preference, value in preference_summary.items():
   print(f"{preference}: {value:.2%}")


# Displaying dietary needs preferences
print("\nDietary Needs Preferences:")
print(dietary_needs_counts)




#To conduct a new survey that outlines and analyzes preferences for a restaurant focusing on local brands, organic ingredients,
# with a delivery service, and options for various dietary needs,
# We will collect preferences on these specific aspects and analyze the data to validate the "Green Spoon" concept.


# Simulate survey data
np.random.seed(42)  # For reproducibility


num_students = 300


# Simulated preferences
preferences = {
   'Student_ID': np.arange(1, num_students + 1),
   'Prefers_Local_Brands': np.random.choice([True, False], num_students, p=[0.7, 0.3]),
   'Seeks_Organic_Ingredients': np.random.choice([True, False], num_students, p=[0.8, 0.2]),
   'Wants_Delivery_Service': np.random.choice([True, False], num_students, p=[0.9, 0.1]),
   'Needs_Dietary_Options': np.random.choice(['None', 'Vegan', 'Gluten-Free', 'Dairy-Free'], num_students, p=[0.2, 0.3, 0.25, 0.25]),
   'Values_Discounts_And_Rewards': np.random.choice([True, False], num_students, p=[0.85, 0.15]),
}




#Step 3: Visualization
# Set the visual style
sns.set(style="whitegrid")


# Visualizing General Preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=list(preference_summary.keys()), y=list(preference_summary.values()), palette="viridis")
plt.title('Student Preferences for "Green Spoon" Concept')
plt.ylabel('Percentage')
plt.xlabel('Preference Categories')
plt.show()


# Visualizing Dietary Needs
plt.figure(figsize=(10, 6))
dietary_needs_counts.plot(kind='bar', color='teal')
plt.title('Student Dietary Needs Preferences')
plt.ylabel('Number of Students')
plt.xlabel('Dietary Need')
plt.xticks(rotation=45)
plt.show()


#Analysis and Suggested Concept Validation:
#High Demand for Organic and Local: The survey data suggest a strong preference for organic ingredients and local brands among students, validating the core values of "Green Spoon."
#Delivery Service is Essential: The overwhelming preference for delivery service underscores the importance of including this feature in the restaurant's offerings.
#Interest in Dietary Options: A significant portion of students have specific dietary needs, highlighting the necessity for "Green Spoon" to offer a variety of menu options catering to vegan, gluten-free, and dairy-free diets.
#Discounts and Rewards Are Valued: The high interest in discounts and rewards suggests that incorporating a loyalty program, such as the proposed stamp card for free desserts, could effectively drive repeat visits.
#Based on this analysis, the "Green Spoon" concept is well-aligned with student preferences, offering a strong foundation for a successful restaurant next to the university. Continuous engagement and feedback collection will be key to refining and adapting the concept to meet evolving student needs.
