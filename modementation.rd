Sommaire :
Contexte :  machine learning in iot , prédiction of occupancy of rooms desktop in building, 
            convert occupancy sensors continous variable into a binary variable of Occupancy/absence
            and hence a classification feature.

+ Data : 
    + Data description : 

        + While we take a desktop, room & kitchen as example, an try to prédict the Occupancy of each ones
        + In the predictable features we have : 
        + The Temperture Outside , wind's speed and direction,
        + The inside Temperture and Humidity sensors 
        + Plug-in used or not 
        + The windows is open or not 
        + Energie consomption  

        in side we 
        we while try 3 models : 
            Model1:  Occupancy of Desktop ~  all features related to it 
    
    + Data Exploration : 
        + unbalanced between occupancy and absence 30 %\ in average, which unbalanced classification class
        + A time serie each 15 min : sensors calculate the time occupied of the total time : we have percent of time occuiped in each room, Desktop & kitchen. 
        + Discritization of continous varibale into binary class {0} in 0 percent Absence else we assume {1} occupancy
        + normalisation of the distribution ? 
        + Correlation : we don't have a significant correlation between the target and other features
        +         

    + Modelisation : 
        + the approch of modélisation 
        + Split into Train & test the dataset
        + machine learning classification : Xgboost as main model presentation
        + metrics definition 
        + Evalution 
        + interpretation 


