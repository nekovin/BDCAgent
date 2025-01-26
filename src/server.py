from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import json
from io import BytesIO
from dotenv import load_dotenv
import os
import openai

from pydantic_ai.models.openai import OpenAIModel
from agents.cleaning_agent import CleaningAgent
from agents.planning_agent import PlanningAgent
from agents.causation_agent import CausationAgent

from collections import defaultdict

from utils import *

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_data = None
cleaning_plan = None
cleaned_data = None

@app.get("/")
async def get():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        global uploaded_data
        uploaded_data = df
        return {"status": "success", "message": f"Uploaded: {len(df)} rows"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
session_context = defaultdict(lambda: {"history": [], "current_plan": None})

def clean_data(cleaning_agent, plan: str, data: pd.DataFrame):
    cleaned_df = cleaning_agent.clean_data(data, plan)
    return cleaned_df.head() #currently returning the head of the data

def analyze_causation(causation_agent, data: pd.DataFrame, plan: str):
    print(type(data))
    print(type(plan))
    res = causation_agent.analyze_causation(data, plan)
    return res
    
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    model = OpenAIModel("gpt-4o-mini")
    planning_agent = PlanningAgent(model)
    cleaning_agent = CleaningAgent(model)
    causal_agent = CausationAgent(model)

    session_id = id(websocket) 
    session = session_context[session_id]

    response = ''

    try:
        await websocket.send_text("## Planning Agent: Hello! How can I assist you today?")
        while True:
            user_input = await websocket.receive_text()
            session["history"].append({"role": "user", "content": user_input})

            try:
                if uploaded_data is not None:
                    
                    response, plans = planning_agent.infer_response(user_input, uploaded_data)
                    
                    #await websocket.send_text(f"Debugging: {response}, \n{plans}")
                    if plans is not None:
                        await websocket.send_text(f"Planning...")
                        await websocket.send_text(f"Plans were made...")
                        session["current_plan"] = plans
                        await websocket.send_text(f"Cleaning...")
                        #await websocket.send_text(f"Type of cleaning: {type(plans['cleaning_plan'])} and type of data: {type(uploaded_data)}")
                        cleaned_data = clean_data(cleaning_agent, plans['cleaning_plan'], uploaded_data)
                        #await websocket.send_text(cleaned_data)
                        await websocket.send_text(f"Analysing...")
                        causal_analysis = analyze_causation(causal_agent, cleaned_data, plans['causal_plan'])
                        await websocket.send_text(causal_analysis)
                    else:
                        await websocket.send_text(f"Planning Agent: {response}")
                else:
                    #await websocket.send_text(f"Data is none.")
                    response, plans = planning_agent.infer_response(user_input, None)
                    await websocket.send_text(f"Planning Agent: {response}")

                session["history"].append({"role": "agent", "content": response})
            except Exception as e:
                response, plans = f"An error occurred: {str(e)}", None
                session["history"].append({"role": "agent", "content": response})
            
            #await websocket.send_text(f"Debugging: {response}")

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
        del session_context[session_id]