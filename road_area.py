import geopandas as gpd
import numpy as np
from tqdm import tqdm

# 실폭도로 데이터 > area of road for each grid cell

roads = gpd.read_file('data/seoul_road/실폭도로/TL_SPRD_RW_11.shp').to_crs(epsg=5174)

grid['road_area'] = np.nan

for i, row in tqdm(grid.iterrows(), total=grid.shape[0]):
    area = roads.intersection(row['geometry']).area.sum()
    grid.at[i, 'road_area'] = area

grid.to_pickle('data/grid_with_roadarea.pkl')