import folium
import pandas as pd
import math
#this function is for fixing the error in data location 
def in_istanbul(row, lat_name, lon_name):
    coordinates = row[lat_name]
    if coordinates is not None and len(coordinates) == 2:
        lat, lon = coordinates
        return 40.8121 <= lat <= 41.3613 and 28.5461 <= lon <= 29.5377
    return False

#this function is for fixing the error in data entry
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


# Load both datasets
df1 = pd.read_excel('bisiklet-ve-mikromobilite-park-alanlar.xlsx')
df2 = pd.read_excel('park-ve-yeil-alan-koordinatlar.xlsx')

# Preprocess coordinates for the second dataset
df2 = df2.dropna(subset=['KOORDİNAT (Yatay , Dikey)'])
koordinat = df2['KOORDİNAT (Yatay , Dikey)']
koordinat = koordinat.apply(preprocess_coordinates)
df2['KOORDİNAT (Yatay , Dikey)'] = koordinat

#wrong coordinates in the data so we clean them


#this part is basically  fixing the error in the data
#these errors accur because of mistakes in coordinates data in park.csv
df2= df2[df2.apply(in_istanbul, args=('KOORDİNAT (Yatay , Dikey)', 'KOORDİNAT (Yatay , Dikey)'), axis=1)]
df2 = df2[df2['TÜR'] == 'PARK']

# Manually correct the coordinates for "ORTA MAHALLE FATİH PARKI"
# Replace 'correct_lat' and 'correct_lon' with the correct latitude and longitude
correct_lat = 41.0  # replace with correct latitude
correct_lon = 29.0  # replace with correct longitude

fatih_park = df2[df2['MAHAL ADI'] == 'ORTA MAHALLE FATİH PARKI'].copy()
fatih_park['KOORDİNAT (Yatay , Dikey)'] = [[correct_lat, correct_lon]] * len(fatih_park)

df2=df2.loc[df2['MAHAL ADI'] != 'ORTA MAHALLE FATİH PARKI']  # select all but FATİH PARKI
df2 = pd.concat([df2, fatih_park])  # put corrected FATİH PARKI back

# Create a map centered around the average coordinates of the parks
avg_lat = df1['Y Koordinatı'].mean()
avg_lon = df1['X Koordinatı'].mean()

m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

# Create different feature groups
fg_bike = folium.FeatureGroup(name='Bike and Micromobility Parks')
fg_green = folium.FeatureGroup(name='Parks and Green Areas')

# Add the parks as markers on the map
# Add the parks as markers on the map
for i, row in df1.iterrows():
    folium.Marker(
        location=[row['Y Koordinatı'], row['X Koordinatı']],
        popup=row['Park Alanı Adı'],
        icon=folium.Icon(color="blue", icon="cloud"),
    ).add_to(fg_bike)

for i, row in df2.iterrows():
    coords = row['KOORDİNAT (Yatay , Dikey)']
    # Ensure there are exactly two values
    if coords and len(coords) == 2:
        lat, lon = coords
        folium.Marker(
            location=[lat, lon],
            popup=row['MAHAL ADI'],
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(fg_green)


# Add feature groups to the map
fg_bike.add_to(m)
fg_green.add_to(m)

# Add layer control to the map
folium.LayerControl().add_to(m)

# Defining the symbol color as their names
legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 120px; height: 90px; 
     border:2px solid grey; z-index:9999; font-size:14px;" id="legend">
     <div style="background-color:blue;color:white;padding:3px;font-size:10px;text-align:center;">Bicycle and Micromobility Parking Areas</div>
     <div style="background-color:red;color:white;padding:3px;font-size:10px;text-align:center;">Parks and Green Spaces</div>
     </div>
     '''


# Add legend to map
m.get_root().html.add_child(folium.Element(legend_html))

# Save the map to an HTML file
m.save('parks.html')
