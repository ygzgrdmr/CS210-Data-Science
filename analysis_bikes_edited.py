import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import numpy as np
# Read the data from the .csv file
df = pd.read_excel('bisiklet-ve-mikromobilite-park-alanlar.xlsx')

# Show the first few rows of the dataframe to check it's read correctly
print(df.head())

# Count the number of each type of park in each district ("İlçe")
district_counts = df.groupby(['İlçe', 'Park Alanı Tipi']).size().unstack(fill_value=0)
district_counts['Toplam'] = district_counts.sum(axis=1)
print(district_counts)

# Count the number of each type of park in each region ("Bölge")
region_counts = df.groupby(['Bölge', 'Park Alanı Tipi']).size().unstack(fill_value=0)
region_counts["Toplam"]=region_counts.sum(axis=1)
print(region_counts)
# Find the mean coordinates of the bike parks
mean_x = df['X Koordinatı'].mean()
mean_y = df['Y Koordinatı'].mean()
print('Mean X coordinate:', mean_x)
print('Mean Y coordinate:', mean_y)




plt.scatter(df["X Koordinatı"], df["Y Koordinatı"])  # Plot park locations on a scatter plot
plt.xlabel("X Koordinatı")
plt.ylabel("Y Koordinatı")
plt.title("Park Locations")
plt.show()

region_counts.plot(kind="bar", stacked=True)
plt.xlabel("Bölge")
plt.ylabel("Park Alanı Sayısı")
plt.title("Park Alanı Tipi Dağılımı")
plt.legend(title="Park Alanı Tipi")
plt.show()

df = pd.get_dummies(df)

X = df[['X Koordinatı', 'Y Koordinatı']]

# Create the dependent variable
y = df['Bölge_Avrupa']  # Assuming you want to predict 'Bölge_Avrupa'

# Perform logistic regression
logreg = LogisticRegression()
logreg.fit(X, y)

# Predict the probabilities of 'Bölge_Avrupa'
y_pred_proba = logreg.predict_proba(X)[:, 1]

# Plot the predicted probabilities and regression line
plt.scatter(df['X Koordinatı'], df['Y Koordinatı'], c=y_pred_proba, cmap='RdYlBu')
plt.colorbar(label='Predicted Probability')
plt.xlabel('X Koordinatı')
plt.ylabel('Y Koordinatı')
plt.title('Logistic Regression - Predicted Probability of Bölge_Avrupa')

# Add regression line
coef = logreg.coef_[0]
intercept = logreg.intercept_

x_values = np.linspace(df['X Koordinatı'].min(), df['X Koordinatı'].max(), 100)
y_values = -(coef[0] * x_values + intercept) / coef[1]

plt.plot(x_values, y_values, color='black', linewidth=2, label='Regression Line')

plt.legend()
plt.show()

X = df[['X Koordinatı', 'Y Koordinatı']]

# Create the dependent variable
y = df['Bölge_Anadolu']  # Assuming you want to predict 'Bölge_Avrupa'

# Perform logistic regression
logreg = LogisticRegression()
logreg.fit(X, y)

# Predict the probabilities of 'Bölge_Avrupa'
y_pred_proba = logreg.predict_proba(X)[:, 1]

# Plot the predicted probabilities and regression line
plt.scatter(df['X Koordinatı'], df['Y Koordinatı'], c=y_pred_proba, cmap='RdYlBu')
plt.colorbar(label='Predicted Probability')
plt.xlabel('X Koordinatı')
plt.ylabel('Y Koordinatı')
plt.title('Logistic Regression - Predicted Probability of Bölge_Anadolu')

# Add regression line
coef = logreg.coef_[0]
intercept = logreg.intercept_

x_values = np.linspace(df['X Koordinatı'].min(), df['X Koordinatı'].max(), 100)
y_values = -(coef[0] * x_values + intercept) / coef[1]

plt.plot(x_values, y_values, color='black', linewidth=2, label='Regression Line')

plt.legend()
plt.show()