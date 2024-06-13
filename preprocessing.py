import geopandas as gpd
import shapely
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import momepy
import networkx as nx
from shapely.geometry import LineString

# Seoul area

selected_GU = ['강남구', '서초구', '동작구', '관악구', '송파구']
selected_GU = ['서울특별시 ' + gu for gu in selected_GU]

seoul = gpd.read_file('data/seoul_geo/seoul.shp').to_crs(epsg=5174)
seoul_selected = seoul[seoul.SGG_NM.isin(selected_GU)]

elevation = gpd.read_file('data/elevation/표고 5000/N3P_F002.shp') # crs 5174
# select intersects with seoul_selected
elevation = gpd.sjoin(elevation, seoul_selected)

speed_long = pd.read_csv('data/traffic/speed_long.csv') # speed data

link = pd.read_excel('data/traffic/link.xlsx')
link = gpd.GeoDataFrame(link, geometry=gpd.points_from_xy(link['GRS80TM_X'], link['GRS80TM_Y']), crs='epsg:5174')

# to int
available_links = speed_long.columns[1:-1].astype(int)

# Select link where speed data is available
link = link[link['LINK_ID'].isin(available_links)]
link_line = link.groupby('LINK_ID').apply(lambda x: LineString(x['geometry'].tolist())).reset_index()
link_line.columns = ['LINK_ID', 'geometry']
link_line = gpd.GeoDataFrame(link_line, crs='epsg:5174')
link_line = gpd.sjoin(link_line, seoul_selected)
print(link_line.shape)

# Momepy
ntw = momepy.gdf_to_nx(link_line)
print('networkx object created')

shortest_path = nx.floyd_warshall_numpy(ntw, weight='mm_len')

np.save('data/traffic/shortest_path.npy', shortest_path)