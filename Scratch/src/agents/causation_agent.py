from langchain.tools import tool
import pandas as pd
from langchain_openai import ChatOpenAI
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

import pandas as pd

@tool
def correlation_tool(params: str) -> str:
    """Calculate correlation between two variables"""
    try:
        file_path = params.split('[')[0].strip()
        vars_str = params.split('[')[1].split(']')[0]
        var1, var2 = [v.strip() for v in vars_str.split(',')]
        
        df = pd.read_csv(file_path)
        correlation = df[var1].corr(df[var2])
        return f"Correlation between {var1} and {var2}: {correlation:.4f}"
    except Exception as e:
        return f"Error calculating correlation: {str(e)}"

class CausationAgent:
    def __init__(self, agent: ChatOpenAI):
        self.agent = agent
        
    def analyze_causation(self, formatted_instruction: str, data : pd.DataFrame) -> Optional[str]:
        try:
            params = formatted_instruction.split("Action Input:")[1].strip()
            
            results = []
            results.append(correlation_tool.invoke(params, data))
            
            return "\n".join(results)
            
        except Exception as e:
            print(f"Error analyzing causation: {e}")
            return None