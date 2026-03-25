import joblib
import pandas as pd
from contextlib import asynccontextmanager
from pathlib import Path
from enum import Enum

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .ml.data import process_data
from .ml.model import inference, compute_model_metrics

MODEL_DIR = Path(__file__).parent.parent / "model"
SLICE_OUTPUT = Path(__file__).parent.parent / "slice_output.txt"
DATA_PATH = Path(__file__).parent.parent / "data" / "census.csv"
CAT_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]


# make sure all necessary files have been generated
@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [
        p for p in [
            MODEL_DIR / "trained_model.pkl",
            MODEL_DIR / "encoder.pkl",
            MODEL_DIR / "lb.pkl",
        ]
        if not p.exists()
    ]
    if missing:
        raise RuntimeError(f"Model files missing: {missing}. Run train_model.py first.")
    yield


app = FastAPI(lifespan=lifespan)


# predifined fields (exemplary for Education, since its a lot of work)
class Education(str, Enum):
    other = "other"
    preschool = "Preschool"
    cl14 = "1st-4th"
    cl56th = "5th-6th"
    cl78th = "7th-8th"
    cl9th = "9th"
    cl10th = "10th"
    cl11th = "11th"
    cl12th = "12th"
    hs_grad = "HS-grad"
    bachelors = "Bachelors"
    masters = "Masters"
    doctorate = "Doctorate"
    some_college = "Some-college"
    assoc_acdm = "Assoc-acdm"
    assoc_voc = "Assoc-voc"
    prof_school = "Prof-school"


EDUCATION_NUM = {
    Education.other: 0,
    Education.preschool: 1,
    Education.cl14: 2,
    Education.cl56th: 3,
    Education.cl78th: 4,
    Education.cl9th: 5,
    Education.cl10th: 6,
    Education.cl11th: 7,
    Education.cl12th: 7,
    Education.hs_grad: 9,
    Education.some_college: 10,
    Education.assoc_voc: 11,
    Education.assoc_acdm: 12,
    Education.bachelors: 13,
    Education.masters: 14,
    Education.prof_school: 15,
    Education.doctorate: 16,
}


# complete user input
class CensusData(BaseModel):
    age: int = 20
    workclass: str = "State-gov"
    fnlgt: int = 123456
    education: Education = Education.bachelors
    marital_status: str = Field("Never-married", alias="marital-status")
    occupation: str = "Adm-clerical"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_gain: int = Field(2174, alias="capital-gain")
    capital_loss: int = Field(0, alias="capital-loss")
    hours_per_week: int = Field(40, alias="hours-per-week")
    native_country: str = Field("United-States", alias="native-country")

    model_config = {"populate_by_name": True}


# methods
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Census Income Prediction API",
        "docs": "/docs",
        "endpoints": ["/metrics", "/slices", "/inference"],
    }


@app.get("/metrics")
async def metrics():
    """Return overall model metrics on the full dataset."""
    model = joblib.load(MODEL_DIR / "trained_model.pkl")
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    lb = joblib.load(MODEL_DIR / "lb.pkl")

    data = pd.read_csv(DATA_PATH, skipinitialspace=True)
    X, y, _, _ = process_data(
        data, categorical_features=CAT_FEATURES, label="salary",
        training=False, encoder=encoder, lb=lb,
    )
    preds = inference(model, X)
    precision, recall, f1 = compute_model_metrics(y, preds)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


@app.get("/slices")
async def slices():
    """Return per-slice metrics from slice_output.txt."""
    if not SLICE_OUTPUT.exists():
        return {"error": "slice_output.txt not found, run training first"}
    return {"slices": SLICE_OUTPUT.read_text()}


@app.post("/inference")
async def predict(data: CensusData):
    """Run model inference on a single census record."""
    model = joblib.load(MODEL_DIR / "trained_model.pkl")
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    lb = joblib.load(MODEL_DIR / "lb.pkl")

    row = data.model_dump(by_alias=True, mode='json')
    row["education-num"] = EDUCATION_NUM[row["education"]]

    # bring correct order, since we appended education
    df = pd.DataFrame([row])[[
        "age", "workclass", "fnlgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
    ]]

    X, _, _, _ = process_data(
        df, categorical_features=CAT_FEATURES, label=None,
        training=False, encoder=encoder, lb=lb,
    )

    preds = inference(model, X)
    salary = lb.inverse_transform(preds.reshape(-1, 1))[0]

    return {"prediction": salary}
