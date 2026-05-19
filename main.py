import os
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from pymongo import MongoClient

app = FastAPI(title="AgriVision Pro AI Backend Engine", version="8.0")

# 🔌 CORS Configuration - Allows Frontend to talk to Backend flawlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 💾 MongoDB Atlas Cluster Connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://mock_user:mock_pass@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority")
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = mongo_client["agrivision_database"]
    telemetry_logs = db["telemetry_logs"]
    print("✅ MongoDB Connected!")
except Exception as e:
    print(f"⚠️ MongoDB running on local fallback: {e}")
    telemetry_logs = None

# 📱 SMS Gateways Configuration
FAST2SMS_API_KEY = os.getenv("FAST2SMS_KEY", "YOUR_FAST2SMS_API_KEY_HERE")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_SID", "YOUR_TWILIO_ACCOUNT_SID_HERE")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_TOKEN", "YOUR_TWILIO_AUTH_TOKEN_HERE")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE", "YOUR_TWILIO_PHONE_NUMBER_HERE")

class SMSPayload(BaseModel):
    mobile: str
    message: str

def send_via_fast2sms(mobile: str, message: str):
    url = "https://www.fast2sms.com/dev/bulkV2"
    headers = {'authorization': FAST2SMS_API_KEY, 'Content-Type': "application/json"}
    payload = {"route": "q", "message": message, "language": "english", "numbers": mobile}
    return requests.post(url, json=payload, headers=headers)

def send_via_twilio(mobile: str, message: str):
    if not mobile.startswith("+"):
        mobile = f"+91{mobile}"
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = {"To": mobile, "From": TWILIO_PHONE_NUMBER, "Body": message}
    return requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))

# --- 🚀 Endpoints ---

@app.post("/send-sms")
async def send_farmer_sms(payload: SMSPayload):
    print(f"📥 Received SMS Request: {payload.mobile} -> {payload.message}")
    
    # If API keys are not provided, run in Sandbox/Simulation Mode automatically
    if "YOUR_" in FAST2SMS_API_KEY and "YOUR_" in TWILIO_ACCOUNT_SID:
        print("💡 Simulation Mode: SMS printed to terminal successfully!")
        return {"status": "SIMULATION_SUCCESS", "message": payload.message}
        
    if "YOUR_" not in FAST2SMS_API_KEY:
        try:
            res = send_via_fast2sms(payload.mobile, payload.message)
            if res.status_code == 200:
                return {"status": "SUCCESS", "gateway": "Fast2SMS"}
        except Exception as e:
            print(f"⚠️ Fast2SMS failed: {e}")

    if "YOUR_" not in TWILIO_ACCOUNT_SID:
        try:
            res = send_via_twilio(payload.mobile, payload.message)
            if res.status_code in [200, 201]:
                return {"status": "SUCCESS", "gateway": "Twilio"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Twilio failed: {str(e)}")

    raise HTTPException(status_code=400, detail="API Key configuration missing.")

@app.post("/predict")
async def predict_crop_disease(file: UploadFile = File(...)):
    print(f"📸 Image analysis triggered for file: {file.filename}")
    time.sleep(1.0) # Processing Delay Simulation
    
    mock_prediction = {
        "class": "Potato Early Blight",
        "disease": "Early Blight",
        "disease_hindi": "(अगेती झुलसा रोग)",
        "confidence": "96.4",
        "cure": "Spraying Copper Oxychloride 50% WP @ 2.5 grams per liter of water is highly recommended."
    }
    
    if telemetry_logs is not None:
        try:
            telemetry_logs.insert_one({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "target_specimen": mock_prediction["class"],
                "confidence_score": f"{mock_prediction['confidence']}%"
            })
        except Exception as e:
            print(f"Database logging error: {e}")
            
    return mock_prediction

@app.get("/")
def check_status():
    return {"status": "ONLINE"}