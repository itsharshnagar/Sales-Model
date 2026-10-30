from fastapi import FastAPI
from pydantic import BaseModel 
import joblib

app = FastAPI()

model = joblib.load("sales.pkl")

class Sales(BaseModel):
    AdvertisingBudget: float

@app.get("/")
def test():
    return {"msg":"uvicorn is running"}

@app.post("/getdata")
def filldata(sl:Sales):
    predict_new = model.predict([[sl.AdvertisingBudget]])
    return {"predicted_sales": round(float(predict_new[0]),2)}

