import os, json, joblib, time
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

MODEL_PATH = os.getenv("MODEL_PATH", "/opt/ml/model/model.joblib")
PREPROC_PATH = os.getenv("PREPROC_PATH", "/opt/ml/model/preproc.joblib")

app = FastAPI()

REQUEST_COUNT = Counter("invocations_total", "Number of invocations")
LATENCY = Histogram("invocation_latency_seconds", "Latency of invocations")

model = None
preproc = None

def load_artifacts():
    global model, preproc
    model = joblib.load(MODEL_PATH)
    preproc = joblib.load(PREPROC_PATH)

@app.on_event("startup")
def startup_event():
    load_artifacts()

@app.get("/ping")
def ping():
    # Health check route
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/invocations")
def invocations(payload: dict):
    '''
    SageMaker-compatible inference.
    Payload: {"instances": [{feature:value,...}, ...]}
    '''
    REQUEST_COUNT.inc()
    start = time.time()

    instances = payload.get("instances", [])
    df = pd.DataFrame(instances)
    Xp = preproc.transform(df)
    proba = model.predict_proba(Xp)[:,1]
    preds = (proba >= 0.5).astype(int).tolist()
    latency = time.time() - start
    LATENCY.observe(latency)
    return {"predictions": preds, "probabilities": proba.tolist(), "latency_s": latency}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))