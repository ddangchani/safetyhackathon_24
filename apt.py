import pandas as pd
from tqdm import tqdm

# 현황 정보

apt = pd.read_excel('data/trashcan/apt.xlsx')

# 도로명 주소 좌표 변환

import geopy

geolocator = geopy.Nominatim(user_agent='South Korea')

def get_lat_lon(address):
    location = geolocator.geocode(address)
    return location.latitude, location.longitude

for i, row in tqdm(trashcan.iterrows(), total=trashcan.shape[0]):
    try:
        lat, lon = get_lat_lon(row['설치위치(도로명 주소)'])
        trashcan.at[i, 'lat'] = lat
        trashcan.at[i, 'lon'] = lon

    except:
        continue

trashcan.to_csv('data/trashcan/trashcan.csv', index=False)