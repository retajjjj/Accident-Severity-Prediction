import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import pickle

#feature engineering
df = pd.read_pickle("../data/interim/cleaned_data.pkl")

#drop unwanted features
df = df.drop(["Accident_Index","Year_x", "Location_Easting_OSGR", "Location_Northing_OSGR", "1st_Road_Number", "2nd_Road_Number", "LSOA_of_Accident_Location", "model", "InScotland", "Police_Force","Was_Vehicle_Left_Hand_Drive", "Vehicle_Leaving_Carriageway", "Number_of_Casualties"], axis=1)
df.info()

#target feature
severity_map = {"Slight": 0, "Serious": 1, "Fatal": 2}
df["Accident_Severity"] = df["Accident_Severity"].map(severity_map)

#make
make_encoding = df.groupby("make")["Accident_Severity"].mean()
df["make"] = df["make"].map(make_encoding)


# Severity rate: how severe accidents are in this district
df["district_severity_rate"] = df["Local_Authority_(District)"].map(
    df.groupby("Local_Authority_(District)")["Accident_Severity"].mean()
)

# accident volume: how many accidents happen in this district
df["district_accident_volume"] = df["Local_Authority_(District)"].map(
    df.groupby("Local_Authority_(District)").size()
)

df = df.drop(columns=["Local_Authority_(District)"])




# Severity rate: how severe accidents are in this district
df["highway_severity_rate"] = df["Local_Authority_(Highway)"].map(
    df.groupby("Local_Authority_(Highway)")["Accident_Severity"].mean()
)

# accident volume: how many accidents happen in this district
df["highway_accident_volume"] = df["Local_Authority_(Highway)"].map(
    df.groupby("Local_Authority_(Highway)").size()
)
df = df.drop(columns=["Local_Authority_(Highway)"])



#road type
road_type_encode = {"Single carriageway": 1,
                    "Dual carriageway": 2,
                    "Roundabout": 0 ,
                    "One way street": 1 ,
                    "Slip road": 2,
                    "Unknown": 1}
df["Road_Type"] = df["Road_Type"].map(road_type_encode)

#Road_Surface_Conditions
df["Road_Surface_Conditions"].value_counts()
df.loc[df["Road_Surface_Conditions"]!= "Dry","Road_Surface_Conditions"] = "Not Dry"
df["is_wet_road"] = pd.get_dummies(df["Road_Surface_Conditions"], drop_first = True, dtype=int)
df = df.drop(columns=["Road_Surface_Conditions"])
df["is_wet_road"].value_counts()


#weather conditions
df["Weather_Conditions"].value_counts()
df.loc[df["Weather_Conditions"]!= "Fine no high winds","Weather_Conditions"] = "Ubnormal"
df["is_ubnormal_weather"] = pd.get_dummies(df["Weather_Conditions"], drop_first = True, dtype=int)
df = df.drop(columns=["Weather_Conditions"])
df["is_ubnormal_weather"].value_counts()

#light condition
df["Light_Conditions"].value_counts()
df.loc[df["Light_Conditions"]!= "Daylight","Light_Conditions"] = "Darkness"
df["is_daylight"] = pd.get_dummies(df["Light_Conditions"], drop_first = True, dtype=int)
df = df.drop(columns=["Light_Conditions"])
df["is_daylight"].value_counts()


#urban or rural
df["Urban_or_Rural_Area"].value_counts()
df["is_urban"] = pd.get_dummies(df["Urban_or_Rural_Area"], drop_first=True, dtype=int)
df["is_urban"].value_counts()
df = df.drop(columns=["Urban_or_Rural_Area"])



#day of week, rush hour
df['is_weekend'] = ((df['Day_of_Week'] == "Saturday") | (df['Day_of_Week'] == "Sunday"))
df["is_weekend"].value_counts()
df = df.drop(columns=["Day_of_Week"])
df["is_weekend"] = df["is_weekend"]*1


df["Time"] = pd.to_datetime(df["Time"])
print(df["Time"].dt.hour)
df["Time"].value_counts()
df['is_rush_hour'] = ( ((df['Time'].dt.hour >= 7) & (df['Time'].dt.hour <= 9) ) |
                      ((df['Time'].dt.hour >= 16) & (df['Time'].dt.hour <= 19) ) )

df["is_rush_hour"] = df["is_rush_hour"]*1
df["is_rush_hour"].value_counts()
df = df.drop(columns=["Time"])

#Age_Band_of_Driver
df["Age_Band_of_Driver"].value_counts()
age_midpoint_map = {
    "0 - 5":   2.5, "6 - 10":  8, "11 - 15": 13, "16 - 20": 18, "21 - 25": 23, "26 - 35": 30, "36 - 45": 40, "46 - 55": 50, "56 - 65": 60, "66 - 75": 70, "Over 75": 80
}

df["age"] = df["Age_Band_of_Driver"].map(age_midpoint_map)
df = df.drop(columns=["Age_Band_of_Driver"])


#Vehicle_Manoeuvre
manoeuvre_map = {
    "Going ahead other":"going_ahead",
    "Going ahead right-hand bend":"going_ahead",
    "Going ahead left-hand bend": "going_ahead",

    "Turning right":"turning",
    "Turning left":"turning",
    "U-turn":"turning",

    "Waiting to go - held up":"stationary",
    "Waiting to turn right": "stationary",
    "Waiting to turn left": "stationary",
    "Parked":"stationary",
    "Slowing or stopping":"stationary",


    "Overtaking moving vehicle - offside": "overtaking",
    "Overtaking static vehicle - offside":"overtaking",
    "Overtaking - nearside":"overtaking",
    
    "Changing lane to right":"lane_change",
    "Changing lane to left":"lane_change",

    "Moving off":"other",
    "Reversing":"other",
    "Data missing or out of range":"unknown",
}

df["manoeuvre_grouped"] = df["Vehicle_Manoeuvre"].map(manoeuvre_map)

manoeuvre_encode = {
    "going_ahead": 0,
    "stationary":  1,
    "turning":     2,
    "lane_change": 3,
    "other":       4,
    "overtaking":  5, 
    "unknown":     -1,
}

df["manoeuvre_encoded"] = df["manoeuvre_grouped"].map(manoeuvre_encode)

df = df.drop(columns=["Vehicle_Manoeuvre", "manoeuvre_grouped"])

#Vehicle_Type
df["Vehicle_Type"].value_counts()
vehicle_type_map = {
    "Car":                                      "car",
    "Taxi/Private hire car":                    "car",

    "Motorcycle over 500cc":                    "motorcycle",
    "Motorcycle 125cc and under":               "motorcycle",
    "Motorcycle 50cc and under":                "motorcycle",
    "Motorcycle over 125cc and up to 500cc":    "motorcycle",
    "Motorcycle - unknown cc":                  "motorcycle",
    "Electric motorcycle":                      "motorcycle",

    "Van / Goods 3.5 tonnes mgw or under":      "light_goods",
    "Goods over 3.5t. and under 7.5t":          "light_goods",

    "Goods 7.5 tonnes mgw and over":            "heavy_goods",
    "Goods vehicle - unknown weight":           "heavy_goods",

    "Bus or coach (17 or more pass seats)":     "passenger",
    "Minibus (8 - 16 passenger seats)":         "passenger",
    
    "Agricultural vehicle":                     "other",
    "Other vehicle":                            "other",
}

df["vehicle_type_grouped"] = df["Vehicle_Type"].map(vehicle_type_map)


vehicle_encode = {
    "car": 0,
    "passenger":1,
    "light_goods":2,
    "heavy_goods":3,
    "other":4,
    "motorcycle":5, 
}

df["Vehicle_Type"] = df["vehicle_type_grouped"].map(vehicle_encode)

df = df.drop(columns=["vehicle_type_grouped"])

#X1st_Point_of_Impact
df["X1st_Point_of_Impact"].value_counts()
point_impact_encode = {
    "Data missing or out of range": -1,
    "Did not impact": 0,
    "Back": 1,
    "Front": 2,
    "Nearside": 3,
    "Offside": 4
}
df["X1st_Point_of_Impact"] = df["X1st_Point_of_Impact"].map(point_impact_encode)



#Sex_of_Driver
gender_encode = {
    "Male": 1,
    "Female":0,
    "Not known":1,
    "Data missing or out of range":1,
     
}
df["is_male"] = df["Sex_of_Driver"].map(gender_encode)
df = df.drop(columns=["Sex_of_Driver"])
df["is_male"].value_counts()


#Propulsion_Code
df["Propulsion_Code"].value_counts()
df.loc[df["Propulsion_Code"]!= "Petrol","Propulsion_Code"] = "No Petrol"

df["is_petrol"] = pd.get_dummies(df["Propulsion_Code"], drop_first=True, dtype=int)
df["is_petrol"].value_counts()
df = df.drop(columns=["Propulsion_Code"])





# Junction Location 
junction_location_map = {
    "Not at or within 20 metres of junction":                       0,  
    "Cleared junction or waiting/parked at junction exit":          1,
    "Entering from slip road":                                      2,
    "Leaving roundabout":                                           2,
    "Entering roundabout":                                          3,
    "Leaving main road":                                            3,
    "Entering main road":                                           3,
    "Mid Junction - on roundabout or on main road":                 4,
    "Approaching junction or waiting/parked at junction approach":  5,  
    "Data missing or out of range":                                -1,
}

df["Junction_Location"] = df["Junction_Location"].map(junction_location_map)


road_class_map = {
    "M":             0, 
    "A":             1,  
    "D":             2,  
    "B":             3,  
    "C":             4,  
    "Unclassified":  5,  
}

df["1st_Road_Class"] = df["1st_Road_Class"].map(road_class_map)

#towing / articulation
towing_map = {
    "No tow/articulation":          0,
    "Caravan":                      1,
    "Other tow":                    1,
    "Single trailer":               1,
    "Double or multiple trailer":   1,
    "Articulated vehicle":          1,
    "Data missing or out of range": 0, 
}

df["is_towing"] = df["Towing_and_Articulation"].map(towing_map)

df = df.drop(columns=["Towing_and_Articulation"])

df.info()
df.corr(numeric_only=True)

#df= df.drop(columns=["Date", "Journey_Purpose_of_Driver","Propulsion_Code", "Towing_and_Articulation" , "Vehicle_Type", "vehicle_type_grouped", "Junction_Detail", "Junction_Control", "Driver_Home_Area_Type"])
#df= df.drop(columns=["Driver_Home_Area_Type"])
#df = df.drop(columns =["Junction_Control"])
#df = df.drop(columns=["Junction_Detail"])

df["Journey_Purpose_of_Driver"].value_counts()
df["Driver_Home_Area_Type"].value_counts()
df["Junction_Control"].value_counts()
df["Junction_Detail"].value_counts()

junction_control_map = {
    "Not at junction or within 20 metres":  0,
    "Authorised person":                    1, 
    "Stop sign":                            2,
    "Auto traffic signal":                  3,
    "Give way or uncontrolled":             4, 
    "Data missing or out of range":        -1,
}
df["junction_control_encoded"] = df["Junction_Control"].map(junction_control_map)


junction_detail_map = {
    "Not at junction or within 20 metres":  0,
    "Slip road":                            1,
    "Mini-roundabout":                      2,
    "Roundabout":                           2,
    "Private drive or entrance":            3,
    "T or staggered junction":              4,
    "Other junction":                       4,
    "Crossroads":                           5, 
    "More than 4 arms (not roundabout)":    6, 
    "Data missing or out of range":        -1,
}
df["junction_detail_encoded"] = df["Junction_Detail"].map(junction_detail_map)


journey_map = {
    "Commuting to/from work":           0,
    "Taking pupil to/from school":      0,
    "Pupil riding to/from school":      0,
    "Other":                            1,
    "Not known":                        1,
    "Other/Not known (2005-10)":        1,
    "Journey as part of work":          2,  # highest risk - professional driving
    "Data missing or out of range":    -1,
}
df["journey_purpose_encoded"] = df["Journey_Purpose_of_Driver"].map(journey_map)

driver_area_map = {
    "Urban area":                       0,
    "Small town":                       1,
    "Rural":                            2,
    "Data missing or out of range":    -1,
}
df["driver_area_encoded"] = df["Driver_Home_Area_Type"].map(driver_area_map)
df = df.drop(columns=[
    "Junction_Control",
    "Junction_Detail", 
    "Journey_Purpose_of_Driver",
    "Driver_Home_Area_Type"
])



df["speed_x_road_risk"] = df["Speed_limit"] * df["Road_Type"]

df["Engine_Capacity_.CC."].describe()

df["engine_size_category"] = pd.cut(
    df["Engine_Capacity_.CC."],
    bins=[0, 1400, 2000, 4000, float("inf")],
    labels=[0, 1, 2, 3]   
).astype("Int64")
df["engine_size_category"] = df["engine_size_category"].fillna(-1)


df["wet_road_speed"] = (
    (df["is_wet_road"] == 1) &
    (df["Speed_limit"] >= 60)
).astype(int)

df["young_driver_night"] = (
    (df["age"] <= 25) &
    (df["is_daylight"] == 0)
).astype(int)

df = df.drop(columns=["Date"])

with open("../data/interim/engineered_data.pkl", "wb") as f:
    pickle.dump(df, f)









  
  
  
  
  


