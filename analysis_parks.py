import pandas as pd
import math

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

""" 

"""
