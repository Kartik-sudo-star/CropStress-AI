#!/usr/bin/env python
"""CropStress AI one-command launcher: self-test + start server + open browser."""
import subprocess, sys, time, webbrowser
from pathlib import Path
ROOT = Path(__file__).parent

print("== CropStress AI launcher ==")
need = [ROOT/"models"/m/"best_model.pkl" for m in ["image_only","sensor_only","multimodal"]]
if not all(p.exists() for p in need):
    print("Models missing -> training (2-3 min)...")
    r = subprocess.run([sys.executable, "scripts/bootstrap_working_model.py"], cwd=ROOT)
    if r.returncode != 0: sys.exit("TRAINING FAILED")
    print("Training done.")
else:
    print("Models found.")

print("Self-test: predict on data/raw/images/HE_000.jpg ...")
t = subprocess.run([sys.executable, "-c",
 "from fastapi.testclient import TestClient\nfrom crop_backend import app\nc=TestClient(app)\n"
 "img=open('data/raw/images/HE_000.jpg','rb').read()\n"
 "r=c.post('/api/predict',files={'image':('leaf.jpg',img,'image/jpeg')},"
 "data={'crop':'wheat','soil_moisture':22,'temperature':36,'humidity':32,'rainfall':0,'light_intensity':1750})\n"
 "d=r.json();print('SELFTEST:',r.status_code,d['prediction'],round(d['confidence']*100,1))\n"
 "assert r.status_code==200"],
 cwd=ROOT, capture_output=True, text=True)
print((t.stdout or "")[-300:]); print(t.stderr[-300:] if t.stderr else "")
if t.returncode != 0: sys.exit("SELF-TEST FAILED")

print("Starting server http://127.0.0.1:8000 ... browser opening ...")
webbrowser.open("http://127.0.0.1:8000/")
subprocess.run([sys.executable, "-m", "uvicorn", "crop_backend:app",
                "--host", "127.0.0.1", "--port", "8000"], cwd=ROOT)
