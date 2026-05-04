import pandas as pd
import numpy as np
import pickle 
import seaborn as sns

df_original = pd.read_pickle("../data/interim/engineered_data.pkl")

df = df_original.copy()
df.info()

sns.histplot(df["1st_Road_Class"])

sns.histplot(df["Latitude"])
df["Latitude"] = np.sqrt(df["Latitude"])


sns.histplot(df["Longitude"])
df["Longitude"] = np.sqrt(df["Longitude"])

sns.histplot(df["Number_of_Vehicles"])

sns.barplot(x="Pedestrian_Crossing-Human_Control", y="Severity", data=df)

sns.histplot(df["Pedestrian_Crossing-Physical_Facilities"])
sns.histplot(df["Road_Type"])
sns.histplot(df["Speed_limit"])
sns.histplot(df["Age_of_Vehicle"])
sns.histplot(df["Engine_Capacity_.CC."])
sns.histplot(df["Junction_Location"])
sns.histplot(df["make"])
sns.histplot(df["Vehicle_Location.Restricted_Lane"])
sns.histplot(df["Vehicle_Reference"])
sns.histplot(df["Vehicle_Type"])
sns.histplot(df["X1st_Point_of_Impact"])
sns.histplot(df["temp"])
sns.histplot(df["district_severity_rate"])
sns.histplot(df["highway_severity_rate"])
sns.histplot(df["district_accident_volume"])
sns.histplot(df["highway_accident_volume"])
sns.histplot(df["is_wet_road"])
sns.histplot(df["is_ubnormal_weather"])
sns.histplot(df["is_daylight"])
sns.histplot(df["is_urban"])
sns.histplot(df["is_weekend"])
sns.histplot(df["age"])
sns.histplot(df["is_rush_hour"])
sns.histplot(df["manoeuvre_encoded"])
sns.histplot(df["is_male"])
sns.histplot(df["is_petrol"])
sns.histplot(df["is_towing"])
sns.histplot(df["junction_control_enocoded"])
sns.histplot(df["junction_detail_encoded"])
sns.histplot(df["journey_purpose_encoded"])
sns.histplot(df["driver_area_encoded"])
sns.histplot(df["speed_x_road_risk"])
sns.histplot(df["engine_size_category"])
sns.histplot(df["wet_road_speed"])
sns.histplot(df["young_driver_night"])


