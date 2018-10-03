# Predicting whether a given pokemon is legendary or not

Given the standard stats of all pokemon, estimate whether a given 
pokemon is legendary or not. This is a binary classification example 
where the value *legenary* field is *0* if the pokemon is not a 
legendary pokemon, and *1* if it is.

### File descriptions
- **EDA.ipynb**: Basic EDA performed in jupyter lab
- **dbloader.py**: Script to load dataset into local Postgresql database
- **test.py**: Actual script where modelling is done using python subprocesses


### Languages and tools used:
- Python v3
- packages mentioned in requirements.txt
- Postgresql
Redis

### Instructions on how to use:
- Setup Postgresql DB
- Run script **dbloader.py** to load data into DB
- Run script **test.py**
