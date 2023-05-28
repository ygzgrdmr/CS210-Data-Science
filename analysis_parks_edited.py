import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import mean_squared_error
# Load the park data from the CSV file into a Pandas DataFrame
df = pd.read_excel('park-ve-yeil-alan-koordinatlar.xlsx')

"""
this part is for fixing the error in the data 
def preprocess_coordinates(coordinates):
    try:
        if coordinates is not None and not (type(coordinates) == float and math.isnan(coordinates)):
            coordinates_split = []
            if "," in coordinates:
                coordinates_split = coordinates.split(',')
                for i in range(len(coordinates_split)):
                    coordinates_split[i] = float(coordinates_split[i].strip())
            elif '\n' in coordinates:
                coordinates_split = coordinates.split('\n')
                for i in range(len(coordinates_split)):
                    coordinates_split[i] = float(coordinates_split[i].strip())
            else:
                coordinates_split = coordinates.strip().split(' ')
                for i in range(len(coordinates_split)):
                    coordinates_split[i] = float(coordinates_split[i].strip())
            return coordinates_split
    except:
        pass
    return None
"""

# Create a map centered around the average coordinates of the parks
df = df.dropna(subset=['KOORDİNAT (Yatay , Dikey)'])
print(df.head())
district_counts = df.groupby(['İLÇE', 'TÜR']).size().unstack(fill_value=0)

# Add a new column with the sum of each row
district_counts['Toplam'] = district_counts.sum(axis=1)
print(district_counts)

# Count the number of each type of space overall
type_counts = df['TÜR'].value_counts()
print(type_counts)


x_values = []
y_values = []
for coord in df['KOORDİNAT (Yatay , Dikey)']:
    coordinates = coord.split(' , ')
    if len(coordinates) == 2:
        x_values.append(float(coordinates[0]))
        y_values.append(float(coordinates[1]))
    else:
        # Assign default values for missing coordinate
        x_values.append(0.0)
        y_values.append(0.0)

# Add X and Y columns to the data dictionary
df['X'] = x_values
df['Y'] = y_values
'''
# Create a DataFrame
df = pd.DataFrame(df)

# Encode the TÜR variable as a binary target
df['TÜR_PARK'] = df['TÜR'].apply(lambda x: 1 if x == 'PARK' else 0)

# Separate the X and Y values and the encoded TÜR variable
X = df[['X', 'Y']]
y = df['TÜR_PARK']

# Perform logistic regression
classifier = LogisticRegression()
classifier.fit(X, y)

# Generate points along the X axis
x_range = np.linspace(df['X'].min(), df['X'].max(), num=100)

# Create input data for prediction
prediction_input = pd.DataFrame({'X': x_range, 'Y': x_range})  # Assuming the same range for both X and Y

# Predict the corresponding probabilities
y_prob = classifier.predict_proba(prediction_input)[:, 1]

# Plot the scatter plot of the actual data
plt.scatter(df['X'], df['Y'], c=y, cmap='bwr', label='Actual Data')

# Plot the decision boundary
plt.contourf(prediction_input['X'].values.reshape(-1, 1), prediction_input['Y'].values.reshape(-1, 1), y_prob.reshape(-1, 1), cmap='bwr', alpha=0.2)

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Logistic Regression')

# Add a legend
plt.legend()

# Show the plot
plt.show()
'''

X = df[['X', 'Y']]
y = df['TÜR']

# Encode the labels if they are not already numeric
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_scaled, y_train)

# Predict
y_pred = classifier.predict(X_test_scaled)

# Evaluate
print(classification_report(y_test, y_pred, target_names=le.inverse_transform(np.unique(y_test))))
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot it using Seaborn
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

df.columns
df['TÜR_PARK'] = df['TÜR'].apply(lambda x: 1 if x == 'PARK' else 0)

df['Park Index'] = df['X'] + df['Y'] + df['TÜR_PARK']


# Prepare data
X = df[['X', 'Y']]
y = df['Park Index']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit model
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train)

# Predict
y_pred = regressor.predict(X_test_scaled)

# Evaluate
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

df.drop("SIRA\nNO", axis=1, inplace=True)



print(df.head())
park_type_counts = df["TÜR"].value_counts()

# Plot the bar plot
plt.figure(figsize=(8, 6))
park_type_counts.plot(kind="bar", color="skyblue")
plt.xlabel("Park Type")
plt.ylabel("Count")
plt.title("Distribution of Park Types")
plt.xticks(rotation=45)
plt.show()

