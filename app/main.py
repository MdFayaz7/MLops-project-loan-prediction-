from fastapi import FastAPI
from src.prediction_logger import log_prediction
from prometheus_fastapi_instrumentator import Instrumentator

from pydantic import BaseModel
from src.pipeline.prediction_pipeline import run_prediction_pipeline
app= FastAPI(title="Loan Prediction API", description="API for predicting loan status")
Instrumentator().instrument(app).expose(app,endpoint="/metrrics")

class LoanInput(BaseModel):
    credit_policy: float
    purpose: str
    int_rate: float
    installment: float
    log_annual_inc: float
    dti: float
    fico: float
    days_with_cr_line: float
    revol_bal: float
    revol_util: float
    inq_last_6mths: float
    delinq_2yrs: float
    pub_rec: float


@app.post("/predict")
def predict(data:LoanInput):
    input_dict={
        "credit.policy":data.credit_policy,
         "purpose": data.purpose,
        "int.rate": data.int_rate,
        "installment": data.installment,
        "log.annual.inc": data.log_annual_inc,
        "dti": data.dti,
        "fico": data.fico,
        "days.with.cr.line": data.days_with_cr_line,
        "revol.bal": data.revol_bal,
        "revol.util": data.revol_util,
        "inq.last.6mths": data.inq_last_6mths,
        "delinq.2yrs": data.delinq_2yrs,
        "pub.rec": data.pub_rec,
    }
    prediction,model_version=run_prediction_pipeline(input_dict)
    log_prediction(input_dict, prediction,model_version)
    return{"prediction":prediction,"label":"Approved" if prediction==0 else "Rejected","model_version":model_version}
