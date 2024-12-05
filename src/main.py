import argparse
from utils.utils import get_api
#from managers.prompt_agents import prompt
from langchain_openai import ChatOpenAI
from managers.agent_manager import *
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Causal Agent")
    parser.add_argument("--file_path", type=str, help="Path to the data file", default=r"C:\Datasets\stocks\stocks\A.csv")
    parser.add_argument("--message", type=str, help="Message to process", default="I need to clean the data")
    args = parser.parse_args()
    return args

def setup(args):
    api_key = get_api()

    if api_key:
        print("API key loaded successfully!")
    else:
        print("Failed to load the API key.")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
    
    file_path = args.file_path
    message = args.message

    print(f"File Path: {file_path}")
    print(f"Message: {message}")

    orchestrator = Orchestrator(llm)

    df = load_data(file_path)  

    return message, orchestrator, df

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def main():
    args = parse_args()

    message, orchestrator, df = setup(args)

    result = orchestrator.process(message, df)

    print(result)

if __name__ == "__main__":
    main()