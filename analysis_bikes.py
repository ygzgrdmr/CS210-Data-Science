import pandas as pd

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
