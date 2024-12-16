import argparse
import pandas as pd
from langchain_openai import ChatOpenAI
#rom langchain.memory import FileChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from utils.utils import get_api
from agents.planning_agent import PlanningAgent
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Causal Agent")
    parser.add_argument("--file_path", type=str, help="Path to the data file", default=r"C:\Datasets\stocks\stocks\A.csv")
    parser.add_argument("--message", type=str, help="Message to process", default="I need to clean the data")
    args = parser.parse_args()
    return args

def setup(args, chat_history):
    api_key = get_api()

    if api_key:
        print("\nAPI key loaded successfully!\n")
    else:
        print("\nFailed to load the API key.\n")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
    
    file_path = args.file_path
    message = args.message

    orchestrator = PlanningAgent(llm, chat_history)

    df = load_data(file_path)  

    return message, orchestrator, df

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data.to_dict()

def main():
    args = parse_args()

    chat_history = FileChatMessageHistory("config/chat_history.json")

    data = load_data(args.file_path)

    message, orchestrator, data = setup(args, chat_history)

    result, message_history = orchestrator.process(message, data)

    if os.path.exists("config/chat_history.json"):
        os.remove("config/chat_history.json")

    if os.path.exists("cleaned_data/cleaned_data.csv"):
        os.remove("cleaned_data/cleaned_data.csv")

if __name__ == "__main__":
    main()