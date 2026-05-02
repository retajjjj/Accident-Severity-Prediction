import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df_original = pd.read_pickle("../data/interim/merged_data.pkl")
df = df_original.copy()

df.info()

for col in df.columns:
    print(df[col].value_counts())
    
for col in df.columns:
    print(f"{col} : {df[col].isna().sum()}")

df["Accident_Severity"].value_counts()

#handle missing values
#delete rows: 2nd_Road_Class, Carriageway_Hazards , Special_Conditions_at_Site, Driver_IMD_Decile, Hit_Object_in_Carriageway, Hit_Object_off_Carriageway, Skidding_and_Overturning , time
df = df.drop(["2nd_Road_Class","Carriageway_Hazards", "Special_Conditions_at_Site", 
              "Driver_IMD_Decile", "Hit_Object_in_Carriageway", "Hit_Object_off_Carriageway", "Skidding_and_Overturning", "time", "Year_y", "Did_Police_Officer_Attend_Scene_of_Accident"], axis =1)

#impute 1st_Road_Number with mean
df["1st_Road_Number"] = df["1st_Road_Number"].fillna(df["1st_Road_Number"].mean())

#impute 2nd_Road_Number with mean
sns.scatterplot(df["2nd_Road_Number"])
df["2nd_Road_Number"] = df["2nd_Road_Number"].fillna(df["2nd_Road_Number"].mean())

#impute latitude
df["Latitude"] = df["Latitude"].fillna(df["Latitude"].mean())

#impute Location_Easting_OSGR
df["Location_Easting_OSGR"] = df["Location_Easting_OSGR"].fillna(df["Location_Easting_OSGR"].mean())

#impute Location_Northing_OSGR
df["Location_Northing_OSGR"] = df["Location_Northing_OSGR"].fillna(df["Location_Northing_OSGR"].mean())

#impute logitude
df["Longitude"] = df["Longitude"].fillna(df["Longitude"].mean())

#impute Pedestrian_Crossing-Physical_Facilities
df["Pedestrian_Crossing-Physical_Facilities"] = df["Pedestrian_Crossing-Physical_Facilities"].fillna(df["Pedestrian_Crossing-Physical_Facilities"].mode()[0])

#impute Pedestrian_Crossing-Physical_Facilities
df["Pedestrian_Crossing-Human_Control"] = df["Pedestrian_Crossing-Human_Control"].fillna(df["Pedestrian_Crossing-Human_Control"].mode()[0])

#age band of driver
df['Age_Band_of_Driver'] = df['Age_Band_of_Driver'].apply(
    lambda x: np.nan if (pd.isna(x) or x == 'Data missing or out of range') else x
)

#impute speedlimit
df["Speed_limit"] = df["Speed_limit"].fillna(df["Speed_limit"].mode()[0])


#impute time
df["Time"] = df["Time"].fillna(df["Time"].mode()[0])

#impute inscotland
df["InScotland"] = df["InScotland"].fillna(df["InScotland"].mode()[0])

#impute temp
grouped = df.groupby(['Date'])["temp"].mean()
mask = df["temp"].isna()
df.loc[mask, "temp"] = df.loc[mask, "Date"].map(grouped)

#remove na
df_imp = df.iloc[:,0:49]

has_na = df_imp.isna().any(axis=1)
#is_slight = df_imp['Accident_Severity'] == 'Slight'
#drop_mask = has_na & is_slight
#df_cleaned = df_imp[~drop_mask]
df_cleaned = df_imp[~has_na]
df_cleaned["Accident_Severity"].value_counts()




df_cleaned.isna().sum()



with open("../data/interim/cleaned_data.pkl", "wb") as f:
    pickle.dump(df_cleaned, f)
