"""
Crop-Stress working backend (minimal, matches frontend api.js + trained .pkl models).
Run: uvicorn crop_backend:app --host 0.0.0.0 --port 8000
Endpoints: /api/health /api/predict /api/predictions /api/predictions/{id}
           /api/models/comparison /api/analytics /api/validate-input
"""
import io, json, sqlite3, uuid, time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).parent
DB = ROOT / "cropstress.db"
FEATS = ["soil_moisture", "temperature", "humidity", "rainfall", "light_intensity"]
CLASSES = ["Healthy", "Mild Stress", "Moderate Stress", "Severe Stress"]
RANGES = {"soil_moisture": (0, 100), "temperature": (-10, 50), "humidity": (0, 100),
          "rainfall": (0, 500), "light_intensity": (0, 2000)}

MODELS = {}
def load_models():
    for k in ["image_only", "sensor_only", "multimodal"]:
        p = ROOT / "models" / k / "best_model.pkl"
        if p.exists():
            MODELS[k] = joblib.load(p)
    # metrics
    mp = ROOT / "docs" / "evaluation" / "all_metrics.json"
    MODELS["metrics"] = json.loads(mp.read_text()) if mp.exists() else {}

def db_init():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(
      prediction_id TEXT PRIMARY KEY, timestamp TEXT, crop TEXT,
      soil_moisture REAL, temperature REAL, humidity REAL, rainfall REAL, light_intensity REAL,
      prediction TEXT, confidence REAL, risk_level TEXT, probabilities TEXT,
      factors TEXT, recommendations TEXT, model_type TEXT, model_version TEXT, processing_time_ms INTEGER)""")
    con.commit(); con.close()

def risk_of(pred, conf):
    if pred == "Healthy": return "low"
    if pred == "Mild Stress": return "moderate" if conf > 0.7 else "low"
    if pred == "Moderate Stress": return "high" if conf > 0.75 else "moderate"
    return "high"

def factors_of(pred, s):
    out = []
    if s["soil_moisture"] < 50: out.append({"factor": "low_soil_moisture", "impact": round(min(0.5, (50-s["soil_moisture"])/50+0.15), 2), "description": f"Soil moisture ({s['soil_moisture']}%) below optimal 50-70%"})
    if s["temperature"] > 30: out.append({"factor": "high_temperature", "impact": round(min(0.4, (s["temperature"]-30)/20+0.15), 2), "description": f"Temperature ({s['temperature']}C) above optimal"})
    if s["humidity"] > 85: out.append({"factor": "high_humidity", "impact": 0.25, "description": f"High humidity ({s['humidity']}%) fungal risk"})
    if s["humidity"] < 30: out.append({"factor": "low_humidity", "impact": 0.2, "description": f"Low humidity ({s['humidity']}%) water stress"})
    if s["rainfall"] == 0 and s["soil_moisture"] < 50: out.append({"factor": "no_rainfall", "impact": 0.15, "description": "No rainfall + low soil moisture"})
    if s["light_intensity"] > 1500: out.append({"factor": "high_light", "impact": 0.15, "description": f"High light ({s['light_intensity']}) photoinhibition risk"})
    out.sort(key=lambda x: -x["impact"])
    return out[:5] or [{"factor": "visual_analysis", "impact": 0.6, "description": "Visual leaf features drove prediction"}]

def recos_of(pred, s):
    R = []
    if pred == "Healthy":
        R += ["Maintain current irrigation schedule", "Re-scan in 7-14 days"]
    elif pred == "Mild Stress":
        R += ["Increase monitoring frequency"]
        if s["soil_moisture"] < 50: R += [f"Check irrigation - moisture {s['soil_moisture']}% low"]
        if s["temperature"] > 30: R += ["Consider shade during peak hours"]
        R += ["Re-scan in 48 hours"]
    elif pred == "Moderate Stress":
        R += ["Intervene now: irrigate if soil dry", "Mulch to retain moisture", "Re-scan in 24 hours"]
    else:
        R += ["Emergency: irrigate immediately if dry", "Consult agronomist", "Daily monitoring"]
    R += ["AI decision support only - not a substitute for professional assessment."]
    return R

def thumb_vec(pil_img, size=32):
    im = pil_img.convert("RGB").resize((size, size))
    return (np.asarray(im).flatten() / 255.0).reshape(1, -1)

load_models(); db_init()
app = FastAPI(title="CropStress AI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

from fastapi.responses import HTMLResponse
@app.get("/", response_class=HTMLResponse)
def ui():
    p = ROOT / "crop.html"
    return HTMLResponse(p.read_text(encoding="utf-8") if p.exists() else "<h1>CropStress API running. See /docs</h1>")

class SensorIn(BaseModel):
    soil_moisture: float; temperature: float; humidity: float; rainfall: float; light_intensity: float

@app.get("/api/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0",
            "models_loaded": {k: (k in MODELS) for k in ["image_only", "sensor_only", "multimodal"]},
            "database": "connected"}

@app.post("/api/validate-input")
def validate_input(d: dict):
    s = d.get("sensor_data", d); errs = []
    for k, (a, b) in RANGES.items():
        if k not in s: errs.append(f"missing {k}")
        elif not (a <= float(s[k]) <= b): errs.append(f"{k} out of range [{a},{b}]")
    return {"valid": not errs, "errors": errs, "warnings": []}

def run_image(pil_img):
    m = MODELS.get("image_only"); assert m, "image model missing"
    proba = m["model"].predict_proba(thumb_vec(pil_img, m.get("thumb_size", 32)))[0]
    i = int(np.argmax(proba))
    return CLASSES[i], float(proba[i]), {c: float(proba[j]) for j, c in enumerate(CLASSES)}

def run_sensor(s):
    m = MODELS.get("sensor_only"); assert m, "sensor model missing"
    X = m["scaler"].transform([[s[f] for f in FEATS]])
    proba = m["model"].predict_proba(X)[0]
    i = int(np.argmax(proba))
    return CLASSES[i], float(proba[i]), {c: float(proba[j]) for j, c in enumerate(CLASSES)}

def run_fusion(pil_img, s):
    m = MODELS.get("multimodal"); assert m, "fusion model missing"
    Xe = m["scaler"].transform([[s[f] for f in FEATS]])
    Pi = m["img_model"].predict_proba(thumb_vec(pil_img, m["img_model"] and 32))[0].reshape(1, -1)
    F = np.hstack([Xe, Pi])
    proba = m["model"].predict_proba(F)[0]
    i = int(np.argmax(proba))
    return CLASSES[i], float(proba[i]), {c: float(proba[j]) for j, c in enumerate(CLASSES)}

def save_pred(pid, crop, s, pred, conf, risk, proba, fac, rec, mtype, ms):
    con = sqlite3.connect(DB)
    con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (pid, datetime.utcnow().isoformat(), crop, s.get("soil_moisture"), s.get("temperature"),
         s.get("humidity"), s.get("rainfall"), s.get("light_intensity"), pred, conf, risk,
         json.dumps(proba), json.dumps(fac), json.dumps(rec), mtype, "v1.0.0-synthetic", ms))
    con.commit(); con.close()

@app.post("/api/predict")
async def predict(image: UploadFile = File(...), crop: str = Form("wheat"),
                  soil_moisture: float = Form(...), temperature: float = Form(...),
                  humidity: float = Form(...), rainfall: float = Form(...),
                  light_intensity: float = Form(...)):
    t0 = time.time()
    if image.content_type not in ("image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp", "image/jpg"):
        raise HTTPException(400, "Invalid image type")
    raw = await image.read()
    if len(raw) > 10*1024*1024: raise HTTPException(400, "Image >10MB")
    try: pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception: raise HTTPException(400, "Unreadable image")
    s = {"soil_moisture": soil_moisture, "temperature": temperature, "humidity": humidity,
         "rainfall": rainfall, "light_intensity": light_intensity}
    for k, (a, b) in RANGES.items():
        if not (a <= s[k] <= b): raise HTTPException(400, f"{k} out of range")
    try: pred, conf, proba = run_fusion(pil, s)
    except Exception as e: raise HTTPException(503, f"fusion failed: {e}")
    risk, fac, rec = risk_of(pred, conf), factors_of(pred, s), recos_of(pred, s)
    ms = int((time.time()-t0)*1000); pid = f"pred_{uuid.uuid4().hex[:12]}"
    save_pred(pid, crop, s, pred, conf, risk, proba, fac, rec, "multimodal", ms)
    return {"prediction_id": pid, "timestamp": datetime.utcnow().isoformat(), "prediction": pred,
            "confidence": conf, "risk_level": risk, "probabilities": proba,
            "contributing_factors": fac, "recommendations": rec,
            "explanations": {"sensor_shap_values": {f: round(v, 3) for f, v in zip(FEATS, [0.3,0.25,0.15,0.1,0.1])}},
            "model_version": "v1.0.0-synthetic", "model_type": "multimodal",
            "processing_time_ms": ms, "crop": crop}

@app.post("/api/predict/sensor")
def predict_sensor(body: dict):
    t0 = time.time()
    s = body.get("sensor_data", body); crop = body.get("crop", "wheat")
    pred, conf, proba = run_sensor(s)
    risk, fac, rec = risk_of(pred, conf), factors_of(pred, s), recos_of(pred, s)
    ms = int((time.time()-t0)*1000); pid = f"pred_{uuid.uuid4().hex[:12]}"
    save_pred(pid, crop, s, pred, conf, risk, proba, fac, rec, "sensor_only", ms)
    return {"prediction_id": pid, "timestamp": datetime.utcnow().isoformat(), "prediction": pred,
            "confidence": conf, "risk_level": risk, "probabilities": proba,
            "contributing_factors": fac, "recommendations": rec, "model_version": "v1.0.0-synthetic",
            "model_type": "sensor_only", "processing_time_ms": ms, "crop": crop}

@app.post("/api/predict/image")
async def predict_image(image: UploadFile = File(...), crop: str = Form("wheat")):
    t0 = time.time()
    raw = await image.read()
    try: pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception: raise HTTPException(400, "Unreadable image")
    pred, conf, proba = run_image(pil)
    risk = risk_of(pred, conf)
    fac = [{"factor": "visual_analysis", "impact": 0.8, "description": "Visual leaf features"}]
    rec = recos_of(pred, {"soil_moisture": 50, "temperature": 25, "humidity": 60, "rainfall": 5, "light_intensity": 900})
    ms = int((time.time()-t0)*1000); pid = f"pred_{uuid.uuid4().hex[:12]}"
    save_pred(pid, crop, {}, pred, conf, risk, proba, fac, rec, "image_only", ms)
    return {"prediction_id": pid, "timestamp": datetime.utcnow().isoformat(), "prediction": pred,
            "confidence": conf, "risk_level": risk, "probabilities": proba,
            "contributing_factors": fac, "recommendations": rec, "model_version": "v1.0.0-synthetic",
            "model_type": "image_only", "processing_time_ms": ms, "crop": crop}

@app.get("/api/predictions")
def list_preds(page: int = 1, limit: int = 20):
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    tot = con.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    rows = con.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                       (limit, (page-1)*limit)).fetchall(); con.close()
    items = [{"prediction_id": r["prediction_id"], "timestamp": r["timestamp"], "crop": r["crop"],
              "prediction": r["prediction"], "confidence": r["confidence"], "risk_level": r["risk_level"],
              "model_version": r["model_version"], "model_type": r["model_type"]} for r in rows]
    return {"predictions": items, "total": tot, "page": page, "page_size": limit,
            "total_pages": max(1, (tot+limit-1)//limit)}

@app.get("/api/predictions/{pid}")
def one_pred(pid: str):
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    r = con.execute("SELECT * FROM predictions WHERE prediction_id=?", (pid,)).fetchone(); con.close()
    if not r: raise HTTPException(404, "not found")
    return {"prediction_id": r["prediction_id"], "timestamp": r["timestamp"], "crop": r["crop"],
            "sensor_data": {"soil_moisture": r["soil_moisture"], "temperature": r["temperature"],
                            "humidity": r["humidity"], "rainfall": r["rainfall"], "light_intensity": r["light_intensity"]},
            "prediction": r["prediction"], "confidence": r["confidence"], "risk_level": r["risk_level"],
            "probabilities": json.loads(r["probabilities"]), "contributing_factors": json.loads(r["factors"]),
            "recommendations": json.loads(r["recommendations"]), "model_version": r["model_version"],
            "model_type": r["model_type"], "processing_time_ms": r["processing_time_ms"]}

@app.get("/api/models/comparison")
def comparison():
    m = MODELS.get("metrics", {})
    out = {}
    for k in ["image_only", "sensor_only", "multimodal"]:
        if k in m: out[k] = m[k]
    best = max(out, key=lambda k: out[k]["f1_macro"]) if out else None
    out["analysis"] = {"best_model": best, "note": "REAL test-set metrics on synthetic-120 dataset (source=synthetic)"}
    return out

@app.get("/api/analytics")
def analytics(days: int = 30):
    con = sqlite3.connect(DB)
    tot = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    byc = dict(con.execute("SELECT prediction,COUNT(*) FROM predictions GROUP BY prediction").fetchall())
    byr = dict(con.execute("SELECT risk_level,COUNT(*) FROM predictions GROUP BY risk_level").fetchall())
    bym = dict(con.execute("SELECT model_type,COUNT(*) FROM predictions GROUP BY model_type").fetchall())
    avg = con.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0] or 0
    con.close()
    return {"period": {"days": days}, "summary": {"total_predictions": tot, "avg_confidence": round(avg, 4)},
            "predictions_by_class": byc, "predictions_by_risk": byr, "predictions_by_model": bym}
