#Algorithm->[Logistic Regression, vSupport Vector Classifier (SVC), XGBoost Classifier]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('C:/Users/vtu21/OneDrive/Desktop/CodeClause.com Internship Projects/Tesla.csv')
df.head()

# Check the shape of the dataframe
df.shape

# Get summary statistics of the data
df.describe()

# Check data types and missing values
df.info()

# Plot the Close price
plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

# Drop the 'Adj Close' column
df = df.drop(['Adj Close'], axis=1)

# Check for missing values again
df.isnull().sum()

# Define the list of features
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Plot distributions and box plots for features
plt.subplots(figsize=(20, 10))

for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.distplot(df[col])
plt.show()

plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.boxplot(df[col])
plt.show()

# Split the 'Date' column into day, month, and year
splitted = df['Date'].str.split('/', expand=True)
df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')

df.head()

# Create a binary column 'is_quarter_end'
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
df.head()

# Calculate the mean of columns grouped by 'year'
data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20, 10))

for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2, 2, i + 1)
    data_grouped[col].plot.bar()
plt.show()

# Calculate the mean of columns grouped by 'is_quarter_end'
df.groupby('is_quarter_end').mean()

# Create new columns 'open-close', 'low-high', and 'target'
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Plot a pie chart of the 'target' variable
plt.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
plt.show()

# Plot a heatmap to check for highly correlated features
plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()

# Define features and target variable
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and validation sets
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

# Create a list of models to train
models = [LogisticRegression(), SVC(
    kernel='poly', probability=True), XGBClassifier()]

# Train and evaluate the models
for i in range(3):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]} : ')
    print('Training AUC-ROC Score : ', metrics.roc_auc_score(
        Y_train, models[i].predict_proba(X_train)[:, 1]))
    print('Validation AUC-ROC Score : ', metrics.roc_auc_score(
        Y_valid, models[i].predict_proba(X_valid)[:, 1]))
    print()

# Plot confusion matrix for the first model
metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()
