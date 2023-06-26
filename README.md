CKD Risk Factors Prediction: Exploratory Data Analysis
This repository contains the Exploratory Data Analysis (EDA) for the Chronic Kidney Disease (CKD) dataset. The goal is to predict CKD risk factors using Machine Learning techniques.

Introduction
The EDA involves data importing, data overview, data preprocessing, descriptive statistics, and data visualization.

Data Importing
We started by importing the necessary libraries and loading the CKD dataset using the pandas read_csv function.

# Import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load the data

df = pd.read_csv('ckd-dataset-v2 (2).csv')

Data Overview
We performed an initial exploration of the dataset by printing out the first few rows and checked the basic information of the dataset such as the number of entries, the data types of each column, and the presence of missing values.

# Check the first few rows of the processed data

print(df.head())

# Print the basic information about the dataset

print(df.info())

Data Preprocessing for EDA
We found that two columns ('sg' and 'grf') contained a mixture of numeric ranges, discrete values, and greater than or equal to values. We created a function to handle these special cases and applied it to 'sg' and 'grf' columns to create new columns 'avg_sg' and 'avg_grf'. We then dropped the original 'sg' and 'grf' columns and converted 'class' column to binary format.

# Define a function to process 'sg' and 'grf' columns

def process_column(col):
if isinstance(col, float):
if pd.isnull(col):
return np.nan
else:
if 'discrete' in col:
return np.nan
elif '-' in col:
return np.mean(list(map(float, col.split(' - '))))
elif '≥' in col:
return float(col[2:])
else:
try:
return float(col)
except:
return np.nan

# Apply the function to 'sg' and 'grf' columns

df['avg_sg'] = df['sg'].apply(process_column)
df['avg_grf'] = df['grf'].apply(process_column)

# Drop the original 'sg' and 'grf' columns

df.drop(['sg', 'grf'], axis=1, inplace=True)

# Convert 'class' column to binary format

df['class'] = df['class'].map({'ckd': 1, 'notckd': 0})

Descriptive Statistics
We used the describe function to obtain descriptive statistics for the numeric columns in the dataset.

# Use the describe function

df.describe()

Data Visualization
We visualized the distribution of CKD and non-CKD patients using a bar plot. This helped us understand the balance of the target classes in our dataset.

# Plot histograms for 'avg_sg' and 'avg_grf' columns

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['avg_sg'].dropna(), bins=30, kde=True)
plt.title('avg_sg Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df['avg_grf'].dropna(), bins=30, kde=True)
plt.title('avg_grf Distribution')

plt.tight_layout()
plt.show()

sns.countplot(x='class', data=df)
plt.title('CKD vs Non-CKD Patients')
plt.xlabel('Groups')
plt.ylabel('Number of Patients')
plt.xticks([0, 1], ['Non-CKD', 'CKD'])
plt.show()

Training and Test sets
We then split the data into training and test sets and visualized the distribution of CKD and non-CKD patients in both sets.
from sklearn.model_selection import train_test_split

# Assume 'class' is the target and rest of the columns are features

X = df.drop('class', axis=1)
y = df['class']

# Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Convert y_train and y_test to DataFrames for easier plotting

y_train_df = pd.DataFrame(y_train, columns=['class'])
y_test_df = pd.DataFrame(y_test, columns=['class'])

# Create subplots

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Count plot for 'class' column in the training set

sns.countplot(x='class', data=y_train_df, ax=axs[0])
axs[0].set_title('CKD vs Non-CKD Patients (Training Set)')
axs[0].set_xlabel('Groups')
axs[0].set_ylabel('Number of Patients')
axs[0].set_xticklabels(['Non-CKD', 'CKD'])

# Count plot for 'class' column in the test set

sns.countplot(x='class', data=y_test_df, ax=axs[1])
axs[1].set_title('CKD vs Non-CKD Patients (Test Set)')
axs[1].set_xlabel('Groups')
axs[1].set_ylabel('Number of Patients')
axs[1].set_xticklabels(['Non-CKD', 'CKD'])

plt.tight_layout()
plt.show()

We hope this analysis will contribute to a deeper understanding of the CKD dataset and aid in the development of effective Machine Learning models for CKD prediction.
