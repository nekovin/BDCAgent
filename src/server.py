from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import json
from io import BytesIO
from dotenv import load_dotenv
import os
import openai

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

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            if uploaded_data is not None:
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"You analyze environmental sensor data with columns: {', '.join(uploaded_data.columns)}"},
                        {"role": "user", "content": message}
                    ]
                )
                await websocket.send_text(response.choices[0].message.content)
            else:
                await websocket.send_text("Upload data first")
    except WebSocketDisconnect:
        print("Client disconnected")