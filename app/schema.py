from pydantic import BaseModel
from typing import Optional

class AccidentInput(BaseModel):
    Age_band_of_driver: Optional[str]
    Sex_of_driver: Optional[str]
    Educational_level: Optional[str]
    Vehicle_driver_relation: Optional[str]
    Driving_experience: Optional[str]
    Type_of_vehicle: Optional[str]
    Area_accident_occured: Optional[str]
    Lanes_or_Medians: Optional[str]
    Types_of_Junction: Optional[str]
    Road_surface_type: Optional[str]
    Road_surface_conditions: Optional[str]
    Light_conditions: Optional[str]
    Weather_conditions: Optional[str]
    Type_of_collision: Optional[str]
    Number_of_casualties: Optional[int]
    Vehicle_movement: Optional[str]
    Pedestrian_movement: Optional[str]
    Cause_of_accident: Optional[str]