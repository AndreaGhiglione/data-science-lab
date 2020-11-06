import pandas as pd
import matplotlib.pyplot as plt
from ny_scatter import NYScatterPlot

def get_id_cell(lon, lat, min_lon, max_lon, min_lat, max_lat, num_of_grids):
    lat_grid = (lat - min_lat) / (max_lat - min_lat) * num_of_grids
    lon_grid = (lon - min_lon) / (max_lon - min_lon) * num_of_grids
    id_cell = lon_grid + lat_grid * num_of_grids
    return id_cell

df = pd.read_csv('pois_all_info',sep='\t',low_memory=False)
municipality_pois = pd.read_csv('ny_municipality_pois_id.csv')

# I select the rows with id in ny_municipality through a mask
mask_id = df['@id'].isin(list(municipality_pois.values))
df = df[mask_id]

print(f'Number of missing values over {len(df)} per column: ')
for column in df.keys():
    num_missing_values = len(df.loc[df[column].isnull(),column])
    print(f'{column}: {num_missing_values}')

categories = ['amenity','shop','public_transport','highway']
threshold = 2  # percentage
frequent_data = {}  # dictionary which will contain the categories df (most frequent)
for category in categories:
    df_grouped = df.groupby(category)
    num_values = len(df[category].dropna())  # Number of not null values
    distributions = {}
    for key,group_df in df_grouped:
        if len(group_df) / num_values * 100 >= threshold:
            distributions[key] = len(group_df)

    tmp_df = pd.DataFrame.from_dict(distributions,orient='index')
    tmp_df.columns = ['Occurrences']  # I give a name to the column
    tmp_df = tmp_df.sort_values(by=['Occurrences'],axis=0,ascending=False)

    tmp_df.plot(kind='bar',title=f'Occurences of {category}',grid=True,legend=False,figsize=(14,5),rot=0)
    plt.show()

    frequent_data[category] = tmp_df

freq_data_amenity = df[df['amenity'].isin(frequent_data['amenity'].index)]
freq_data_shop = df[df['shop'].isin(frequent_data['shop'].index)]
freq_data_public_transport = df[df['public_transport'].isin(frequent_data['public_transport'].index)]
freq_data_highway = df[df['highway'].isin(frequent_data['highway'].index)]

freq_data = pd.concat((freq_data_amenity,freq_data_shop,freq_data_public_transport,freq_data_highway))

freq_poi = {'amenity': freq_data_amenity, 'shop': freq_data_shop, 'public_transport': freq_data_public_transport, 'highway': freq_data_highway}  # 'poi' = 'data'

ny_scat_plot = NYScatterPlot()
ny_scat_plot.scatter_plot(freq_poi,'New_York_City_Map.PNG')

(min_lon,max_lon) = (ny_scat_plot.ext[0],ny_scat_plot.ext[1])
(min_lat,max_lat) = (ny_scat_plot.ext[2],ny_scat_plot.ext[3])
num_of_grids = 10
freq_data["grid_id"] = get_id_cell(freq_data["@lon"], freq_data["@lat"], min_lon, max_lon, min_lat, max_lat, num_of_grids)
freq_data["grid_id"] = freq_data["grid_id"].astype(int)

print(freq_data.groupby(['grid_id']).count().drop(columns=['@id','@type','@lat','@lon','name']))
