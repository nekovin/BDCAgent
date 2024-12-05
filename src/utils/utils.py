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
    
def get_api():
    from dotenv import load_dotenv
    import os

    load_dotenv()
    api_key = os.getenv('API_KEY')
    return api_key
    
