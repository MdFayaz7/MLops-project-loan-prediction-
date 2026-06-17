# pyrefly: ignore [missing-import]
from fastapi import FastAPI,Request,HTTPException, status
from fastapi.middleware.cors import CORSMiddleware  # 1. Import CORS Middleware

# pyrefly: ignore [missing-import]
from fastapi.templating import Jinja2Templates

# pyrefly: ignore [missing-import]
from fastapi.responses import HTMLResponse
from src.prediction_logger import log_prediction
# pyrefly: ignore [missing-import]
from prometheus_fastapi_instrumentator import Instrumentator

from pydantic import BaseModel
from src.pipeline.prediction_pipeline import run_prediction_pipeline
from src.components.model_monitoring import ModelMonitoring

app= FastAPI(title="Loan Prediction API", description="API for predicting loan status")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (essential for Hugging Face iframes)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all HTTP headers
)
templates=Jinja2Templates(directory="templates")
Instrumentator().instrument(app).expose(app,endpoint="/metrics")

@app.get("/",response_class=HTMLResponse)
def read_root(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.get("/monitoring", response_class=HTMLResponse)
def get_monitoring_report(refresh: bool = False):
    monitoring = ModelMonitoring()
    # If refresh is true or the report does not exist, trigger a run
    if refresh or not monitoring.report_html_path.exists():
        monitoring.run()
    
    if not monitoring.report_html_path.exists():
        return HTMLResponse(
            content="<h1>Monitoring Report not found. Ensure there is enough prediction log data.</h1>",
            status_code=404
        )
        
    with open(monitoring.report_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


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
    errors=[]

    if data.fico < 300 or data.fico > 850  :
       errors.append("FICO score must be between 300 and 850")
    if data.log_annual_inc <=0 :
       errors.append("annual income should be greater than 0")
    if data.installment <=0 :
        errors.append("installment should be greater than 0")
    if data.int_rate <=0 :
        errors.append("interest rate should be greater than 0")
    
    if errors :
        raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"messages":"Incomplete applictaion data provided."}
           )


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
