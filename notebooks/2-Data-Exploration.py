import pandas as pd
import numpy as np

df_original = pd.read_pickle("../data/interim/merged_data.pkl")
df = df_original.copy()

df.head()

mapping_road_classes = {"A":"A" , "B": "B", "C":"C","Unclassified": "D","Motorway":"M", "A(M)":"M" }
df["1st_Road_Class"] = df["1st_Road_Class"].map(mapping_road_classes)
df["1st_Road_Class"].value_counts()

df["1st_Road_Number"].value_counts()

df["2nd_Road_Class"].value_counts()
df["2nd_Road_Class"] = df["2nd_Road_Class"].map(mapping_road_classes)

df["Accident_Severity"].value_counts()
df["Carriageway_Hazards"].value_counts()

mapping_carraigeway_hazards = {"Other object on road" : "Object", 
                               "Any animal in carriageway (except ridden horse)": "Animal",
                               "Previous accident": "Animal",
                               "Pedestrian in carriageway - not injured": "Pedestrian",
                               "Vehicle load on road": "Vehicle",
                               "Data missing or out of range": None}

df["Carriageway_Hazards"] = df["Carriageway_Hazards"].map(mapping_carraigeway_hazards)

df["Date"].value_counts()

df["Day_of_Week"].value_counts()

df['hour_of_day'] = df['Date'].dt.hour
df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['is_weekend'] = ((df['day_of_week'] == 5) | (df['day_of_week'] == 6)).astype(int)
df['month'] = df['Date'].dt.month

# Map months to seasons
season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
df['season'] = df['month'].map(season_map)