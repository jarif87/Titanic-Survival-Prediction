import pickle
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Load the trained model
with open("ada_model_1.pkl", "rb") as file:
    model = pickle.load(file)

templates = Jinja2Templates(directory="templates")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(request: Request, Pclass: int = Form(...), Sex: int = Form(...), Age: float = Form(...), SibSp: int = Form(...), Parch: int = Form(...), Fare: float = Form(...), Cabin: float = Form(...), Embarked: int = Form(...)):
    features = [Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked]
    prediction = model.predict([features])[0]
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})