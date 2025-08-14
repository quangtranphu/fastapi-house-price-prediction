import os

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from loguru import logger
from schemas import *

# Creating FastAPI instance
app = FastAPI()
# Creating class to define the request body
# and the type hints of each attribute



# Loading model with default path models/model.pkl
clf = joblib.load("./models/model.pkl")

# Creating an endpoint to receive the data
# to make prediction on.
@app.post("/predict")
def predict(data: HouseInfo):
    # Predicting the class
    logger.info("Make predictions...")
    # Convert data to pandas DataFrame and make predictions
    price = clf.predict(pd.DataFrame(jsonable_encoder(data), index=[0]))[0]

    # Return the result
    return {"price": price}
