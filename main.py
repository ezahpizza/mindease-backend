# Standard Library Imports
import os
import json
from datetime import datetime
from typing import Optional
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer

# Third-Party Imports
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from peft import PeftModel
import uvicorn
from fastapi import FastAPI, HTTPException, Header,  Response, Request
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from huggingface_hub import login
from dotenv import load_dotenv
from bson import ObjectId

# Load environment variables
load_dotenv()

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def jsonable_encoder_custom(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: jsonable_encoder_custom(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable_encoder_custom(i) for i in obj]
    return obj

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Prediction API",
    description="API for mental health prediction and therapy chatbot",
    version="1.0.0",
    json_encoder=JSONEncoder
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

HF_TOKEN = os.getenv("HF_TOKEN")

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL")
db_client = AsyncIOMotorClient(MONGODB_URL)
try:
    db_client.admin.command('ping')
    print("ping successful")
except Exception as e:
    print(e)

database = db_client.mental_health_db

class ChatMessage(BaseModel):
    message: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatResponse(BaseModel):
    response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TherapyChatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model_name = "microsoft/phi-2"
        self.peft_model_name = "ezahpizza/mindease-phi"
        self.hf_token = HF_TOKEN
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = None
        self.model = None

    async def load_model(self):
        if self.model is None:
            try:
                print(f"Loading model on device: {self.device}")
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=self.torch_dtype,
                    token=self.hf_token
                ).to(self.device)  # Move base model to device

                # Load PEFT adapted model
                self.model = PeftModel.from_pretrained(
                    base_model,
                    self.peft_model_name,
                    token=self.hf_token
                ).to(self.device)  # Move PEFT model to device

                # Load and configure tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.peft_model_name,
                    token=self.hf_token
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("Model loaded successfully!")

            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise

    async def generate_response(self, message: str) -> str:
        await self.load_model()

        input_text = f"<|context|>{message}<|response|>"

        try:
            # Create inputs and move to the same device as model
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt"
            )
            # Move input tensors to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=5,
                    do_sample=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50
                )

            generated_text = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            response = generated_text.split("<|response|>")[-1].strip()
            return response

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now."
        
# Initialize chatbot instance
chatbot = TherapyChatbot()

# Helper function to verify Clerk user ID
async def verify_user_id(user_id: str = Header(...)):
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="User ID is required"
        )
    return user_id

# Input data models
class PredictionInput(BaseModel):
    user_id: str  # Added user_id to the input model
    gender: int
    age: int
    city: int
    profession: int
    academic_pressure: float
    work_pressure: float
    cgpa: float
    study_satisfaction: float
    job_satisfaction: float
    sleep_duration: int
    dietary_habits: int
    degree: int
    suicidal_thoughts: int
    work_study_hours: float
    financial_stress: float
    mi_history: int

class PredictionResult(BaseModel):
    prediction: float
    prediction_label: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Load ML model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.json')

try:
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
except xgb.core.XGBoostError as e:
    print(f"Error loading model from {MODEL_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Model directory: {BASE_DIR}")
    raise Exception(f"Failed to load XGBoost model: {str(e)}")

# Simplified API endpoints
@app.post("/predict")
async def predict(input_data: PredictionInput):
    user_id = input_data.user_id
    input_dict = {
        'gender': [input_data.gender],
        'age': [input_data.age],
        'city': [input_data.city],
        'profession': [input_data.profession],
        'academic_pressure': [input_data.academic_pressure],
        'work_pressure': [input_data.work_pressure],
        'cgpa': [input_data.cgpa],
        'study_satisfaction': [input_data.study_satisfaction],
        'job_satisfaction': [input_data.job_satisfaction],
        'sleep_duration': [input_data.sleep_duration],
        'dietary_habits': [input_data.dietary_habits],
        'degree': [input_data.degree],
        'suicidal_thoughts': [input_data.suicidal_thoughts],
        'work_study_hours': [input_data.work_study_hours],
        'financial_stress': [input_data.financial_stress],
        'mi_history': [input_data.mi_history]
    }

    input_df = pd.DataFrame(input_dict)
    dmatrix = xgb.DMatrix(input_df)
    
    try:
        prediction = model.predict(dmatrix)[0]
        prediction_label = "High risk of depression" if prediction > 0.5 else "Low risk of depression"
        
        result = PredictionResult(
            prediction=float(prediction),
            prediction_label=prediction_label
        )
        
        # Save prediction to database with user_id
        prediction_doc = {
            "user_id": user_id,
            "input_data": jsonable_encoder_custom(input_dict),
            "prediction": float(result.prediction),
            "prediction_label": result.prediction_label,
            "timestamp": datetime.utcnow()
        }
    
        await database.predictions.insert_one(prediction_doc)
        return jsonable_encoder_custom(result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
    
@app.get("/user/predictions/{user_id}")
async def get_user_predictions(user_id: str):
    try:
        predictions = await database.predictions.find(
            {"user_id": user_id}
        ).to_list(length=100)
        
        return jsonable_encoder_custom(predictions)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching predictions: {str(e)}"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_input: ChatMessage):
    try:
        response = await chatbot.generate_response(chat_input.message)
        
        # Save conversation to MongoDB
        await database.chat_messages.insert_many([
            {
                "user_id": chat_input.user_id,
                "content": chat_input.message,
                "type": "user",
                "timestamp": datetime.utcnow()
            },
            {
                "user_id": chat_input.user_id,
                "content": response,
                "type": "bot",
                "timestamp": datetime.utcnow()
            }
        ])
        
        return ChatResponse(response=response)
    
    except Exception as e:
        print(f"Detailed error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat error: {str(e)}"
        )

@app.get("/chat/history/{user_id}")
async def get_chat_history(user_id: str):
    try:
        messages = await database.chat_messages.find(
            {"user_id": user_id}
        ).sort("timestamp", 1).to_list(length=100)
        
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Startup and shutdown events
@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(MONGODB_URL)
    app.mongodb = app.mongodb_client.mental_health_db
    
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token)
        print("Successfully authenticated with Hugging Face Hub")

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()
