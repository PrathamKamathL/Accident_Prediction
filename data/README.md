Accident Severity Prediction

1.	Input and Output Specification:  
a.	Input: Details of the accident, with a total of 18 features.  
b.	Output: Multiclass classification of severity into three classes:  
i.	Fatal injury – class 1  
ii.	Serious injury – class 2  
iii.	Slight injury – class 3  
2.	Criteria for Success:
a.	Business objective: Minimize the risk of underestimation of accident severity by 10% to improve emergency medical response.  
b.	ML goal: Achieving a recall of 95% for Fatal and Serious injury classes, and maintain an overall macro average precision and F1-score of 90% each.  
3.	Alignment with Business Objectives:   Correctly classifying accident severities can aid in providing appropriate level of immediate medical assistance following the accident.  
4.	Source of the Dataset: https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents , referenced by: https://data.mendeley.com/datasets/xytv86278f/1 
5.	Dataset Description:    
a.	Meaning and Context:
## Feature Relevance to Business Goal

| Feature | Relevance to Business Goal |
|--------|----------------------------|
| Age band of driver | Certain age groups may be associated with more severe accidents due to reaction time or risk-taking behaviour. |
| Gender of driver | May help identify behavioural patterns in driving that may influence accident severity. |
| Education level | Shows awareness of traffic rules and safety practices, which can impact accident outcomes. |
| Vehicle driver relationship | Indicates familiarity with the vehicle and tendency to take responsibility for its care and condition, which may reflect in driving style. |
| Driving experience | Less experienced drivers are more likely to be involved in severe accidents due to poor judgment or control. |
| Type of vehicle | Heavier or high-speed vehicles are more likely to result in severe or fatal injuries. |
| Area of accident occurrence | Urban vs rural areas affect emergency response time and accident severity. |
| Lanes or medians | Road design impacts collision chances and severity. |
| Type of junction where accident occurred | Intersections are high-risk zones where complex traffic movement can lead to severe accidents. |
| Road surface type | Road surface may affect how vehicles behave when drivers lose control, as well as extent of injury on impact with the road. |
| Road surface conditions | Wet roads can make surfaces slippery, increasing severity of accidents. |
| Light conditions | Light conditions affect road visibility. |
| Weather conditions | Adverse weather like rain or snow reduces visibility and control, increasing severity risk. |
| Type of collision | Directly indicates impact intensity and extent of injury. |
| Number of casualties | Higher casualties often correlate with higher accident severity. |
| Vehicle movement | Speed and direction of movement influence impact force and resulting injury severity. |
| Pedestrian movement | Presence and behaviour of pedestrians increase the risk of severe or fatal outcomes. |
Cause of accident | Identifies root factors like over-speeding or distraction, which are strongly linked to severity levels.
b.	Collection Methodology: The dataset was created from Addis Ababa Sub city police departments and has been prepared from manual records of road traffic accidents of years 2017-20. [Mentioned in the reference to dataset]  
c.	Data Partitioning: The chosen dataset contains a total of 12317 samples. They are split into sets as: training set 70% (8621), validation set 10% (1231), test set 20% (2465) samples respectively.   
d.	Preprocessing Logs: (Initially 32 features, removed 14 features)
i.	Features removed due to majority of values being blank/Unknown/having more than 80% of values of one category: defect of vehicle, road alignment, number of vehicles involved, casualty severity, service year of vehicle, age band of casualty, fitness of casualty, work of casualty
ii.	Features removed due to irrelevance to business and ML goal: Accident index, time, day of the week, owner of vehicle, casualty class, sex of casualty  
e.	Technical Specifications:
i.	Storage format: CSV file  
ii.	Number of examples: 12317 in total  

## Feature Allowed Values

| Feature | Allowed Values |
|--------|----------------|
| Age band of driver | Under 18, 18-30, 31-50, Over 51, Unknown |
| Gender of driver | Male, Female, Unknown |
| Education level | High School, Above High School, Junior High School, Unknown |
| Vehicle driver relationship | Employee, Owner, Unknown, Other |
| Driving experience | Below 1yr, 1-2yr, 2-5yr, 5-10yr, Above 10yr |
| Type of vehicle | Automobile, Public, Lorry, Taxi, Station wagon, Pickup, Motorcycle, Bajaj, Ridden horse, Long lorry, Other |
| Area of accident occurrence | Residential areas, Office areas, Recreational areas, Industrial areas, Market areas, Church areas, School areas, Hospital areas, Rural village areas, Outside rural areas, Other, Unknown |
| Lanes or medians | One way, Double carriageway, Two way, Undivided two way, Other |
| Type of junction where accident occurred | Y shape, Crossing, No junction, Unknown, Other |
| Road surface type | Asphalt roads, Earth roads, Gravel roads, Unknown, Other |
| Road surface conditions | Dry, Wet or damp |
| Light conditions | Daylight, Darkness – lights lit, Darkness – no lighting, Darkness – lights unlit |
| Weather conditions | Normal, Raining, Cloudy, Unknown, Other |
| Type of collision | Vehicle with vehicle collision, Collision with roadside objects, Collision with pedestrians, Rollover, Collision with animals |
| Number of casualties | Range: 1 to 8 |
| Vehicle movement | Going straight, Moving backward, Reversing, Turnover, Other |
| Pedestrian movement | Not a pedestrian, Crossing from nearside, Unknown or other, Walking from driver's nearside, Crossing from offside |
| Cause of accident | No distancing, Changing lane to the right, Changing lane to the left, Driving carelessly, No priority to vehicle |