import os
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from pymongo import MongoClient

app = FastAPI(title="AgriVision Pro AI Backend Engine", version="7.2")

# 1. CORS CONFIGURATION (Frontend to Backend connection secure karne ke liye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. MONGODB ATLAS CONNECTION SETUP
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://mock_user:mock_pass@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority")
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = mongo_client["agrivision_database"]
    telemetry_logs = db["telemetry_logs"]
    print("✅ Connected successfully to MongoDB Atlas.")
except Exception as e:
    print(f"⚠️ MongoDB Fallback Active: {e}")
    telemetry_logs = None

# 3. SMS GATEWAY CONFIGURATION CREDENTIALS
FAST2SMS_API_KEY = os.getenv("FAST2SMS_KEY", "YOUR_FAST2SMS_API_KEY_HERE")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_SID", "YOUR_TWILIO_ACCOUNT_SID_HERE")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_TOKEN", "YOUR_TWILIO_AUTH_TOKEN_HERE")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE", "YOUR_TWILIO_PHONE_NUMBER_HERE")

# Pydantic Model (Aapki syntax error yahan fix ho gayi hai)
class SMSPayload(BaseModel):
    mobile: str
    message: str

# FAST2SMS GATEWAY (India Route)
def send_via_fast2sms(mobile: str, message: str):
    url = "https://www.fast2sms.com/dev/bulkV2"
    headers = {
        'authorization': FAST2SMS_API_KEY,
        'Content-Type': "application/json"
    }
    payload = {
        "route": "q",
        "message": message,
        "language": "english",
        "numbers": mobile
    }
    return requests.post(url, json=payload, headers=headers)

# TWILIO GATEWAY (International Route)
def send_via_twilio(mobile: str, message: str):
    if not mobile.startswith("+"):
        mobile = f"+91{mobile}"
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = {
        "To": mobile,
        "From": TWILIO_PHONE_NUMBER,
        "Body": message
    }
    return requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))


# --- 🚀 API ENDPOINTS ---

# 📱 1. LIVE SMS BROADCAST ENDPOINT
@app.post("/send-sms")
async def send_farmer_sms(payload: SMSPayload):
    print(f"📥 SMS Request Recieved: {payload.mobile} -> {payload.message}")
    
    # Sandbox/Simulation mode agar aapke paas abhi keys nahi hain
    if "YOUR_" in FAST2SMS_API_KEY and "YOUR_" in TWILIO_ACCOUNT_SID:
        print("💡 Simulation Mode: SMS sent successfully (Sandbox Log).")
        return {
            "status": "SIMULATION_SUCCESS",
            "info": "Replace dummy keys with real API keys to send live carrier network SMS.",
            "message_preview": payload.message
        }
        
    # Fast2SMS Execute
    if "YOUR_" not in FAST2SMS_API_KEY:
        try:
            res = send_via_fast2sms(payload.mobile, payload.message)
            if res.status_code == 200:
                return {"status": "SUCCESS", "gateway": "Fast2SMS"}
        except Exception as e:
            print(f"⚠️ Fast2SMS failed, trying Twilio: {e}")

    # Twilio Execute
    if "YOUR_" not in TWILIO_ACCOUNT_SID:
        try:
            res = send_via_twilio(payload.mobile, payload.message)
            if res.status_code in [200, 201]:
                return {"status": "SUCCESS", "gateway": "Twilio"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gateways failed: {str(e)}")

    raise HTTPException(status_code=400, detail="Invalid API Key Configuration.")


# 🧠 2. AI CROP INFECTED SPECIMEN INFERENCE ENDPOINT
@app.post("/predict")
async def predict_crop_disease(file: UploadFile = File(...)):
    print(f"📸 Scanning Image: {file.filename}")
    time.sleep(1.5)  # AI model latency processing simulation
    
    mock_prediction = {
        "class": "Potato Early Blight",
        "disease": "Early Blight",
        "disease_hindi": "(अगेती झुलसा रोग)",
        "confidence": "96.4",
        "cure": "Faslon par copper oxychloride 50% WP @ 2.5 gram per liter pani me milakar safe chidkaw karein."
    }
    
    if telemetry_logs is not None:
        try:
            telemetry_logs.insert_one({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "target_specimen": mock_prediction["class"],
                "confidence_score": f"{mock_prediction['confidence']}%",
                "system_status": "COMMITTED"
            })
        except Exception as ex:
            print(f"⚠️ MongoDB write skipped: {ex}")
            
    return mock_prediction


# 🖥️ 3. CENTRAL ROOT NODE STATUS
@app.get("/")
def central_root_node_status():
    return {"status": "ONLINE", "message": "AgriVision Production Server Core Engine is fully deployed."}