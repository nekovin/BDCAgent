from langchain_community.chat_models import ChatOpenAI
from typing import List, Dict
from langchain.tools import tool
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import io
import sys
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import os
from langchain_openai import ChatOpenAI

import argparse

from planning_agent import *
from cleaning_agent import *
from causation_agent import *
from interpretation_agent import *

def load_api_key(file_path):
    """Load the API key from a text file."""
    try:
        with open(file_path, "r") as file:
            api_key = file.read().strip() 
            return api_key
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading API key: {e}")
        return None

def prompt(llm, message, file_path):
    # define agents, planning agent should dynamically adjust which agentss are needed
    planning_agent = PlanningAgent(agent=llm)
    cleaning_agent = CleaningAgent(agent=llm)
    causation_agent = CausationAgent(agent=llm)
    interpretation_agent = InterpretationAgent(agent=llm)

    agent_calls = {
        "CleanData": cleaning_agent.clean_data,
        "FindCorrelation": causation_agent.analyze_causation,
        "Interpret": interpretation_agent.interpret_results
    }

    # split the message into tasks, this needs to be improved
    messages = ["I need to clean the data", "I need to find the correlation between Open and Close", "Interpret"]

    for message in messages:
        print(f"User Message: {message}")
        message_history, response = planning_agent.send_message(message)
        formatted_response = response.replace("[file_path]", file_path)
        #print(f"Planning Agent Response:\n{formatted_response}\n")

        if message == "Interpret":
            print("Interpretation Agent Response:")
            #print(result)
            interpretation = agent_calls["Interpret"](result)
            return interpretation

        if "Action:" in formatted_response:
            action = formatted_response.split("Action:")[1].split("\n")[0].strip()
            if action in agent_calls:
                result = agent_calls[action](formatted_response)
        
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Causal Agent")
    parser.add_argument("--file_path", type=str, help="Path to the data file", default=r"C:\Datasets\stocks\stocks\A.csv")
    parser.add_argument("--message", type=str, help="Message to process", default="I need to clean the data and I need to find the correlation")
    args = parser.parse_args()
    return args    

def main():

    api_key_file = "api_key.txt"  # Path to your API key file
    openai_api_key = load_api_key(api_key_file)

    if openai_api_key:
        print("API key loaded successfully!")
    else:
        print("Failed to load the API key.")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    args = parse_args()

    file_path = args.file_path
    print(f"File Path: {file_path}")

    message = args.message
    print(f"Message: {message}")

    result = prompt(llm, message, file_path)

    print(result)

if __name__ == "__main__":
    main()