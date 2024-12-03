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

@tool
def granger_causality_tool(params: str) -> str:
    """Test for Granger causality between two variables"""
    try:
        file_path = params.split('[')[0].strip()
        vars_str = params.split('[')[1].split(']')[0]
        var1, var2 = [v.strip() for v in vars_str.split(',')]
        
        df = pd.read_csv(file_path)
        data = pd.DataFrame({var1: df[var1], var2: df[var2]})
        
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            result = grangercausalitytests(data, maxlag=5)
            detailed_results = []
            
            for lag in range(1, 6):
                test_results = result[lag][0]
                detailed_results.append(
                    f"Lag {lag}:\n"
                    f"F-test: F={test_results['ssr_ftest'][0]:.4f}, p={test_results['ssr_ftest'][1]:.4f}\n"
                    f"Chi-squared test: chi2={test_results['ssr_chi2test'][0]:.4f}, p={test_results['ssr_chi2test'][1]:.4f}\n"
                    f"Likelihood ratio test: chi2={test_results['lrtest'][0]:.4f}, p={test_results['lrtest'][1]:.4f}\n"
                    f"Parameter F-test: F={test_results['params_ftest'][0]:.4f}, p={test_results['params_ftest'][1]:.4f}\n"
                )
        finally:
            sys.stdout = old_stdout
        
        return "\n".join(detailed_results)
    except Exception as e:
        return f"Error testing Granger causality: {str(e)}"

class CausationAgent:
    def __init__(self, agent: ChatOpenAI):
        self.agent = agent
        
    def analyze_causation(self, formatted_instruction: str):
        try:
            #print("")
            #print("Analyzing causation")
            # Extract file path and variables from instruction
            params = formatted_instruction.split("Action Input:")[1].strip()
            
            # Run analysis tools
            results = []
            #results.append(correlation_tool(params))
            results.append(correlation_tool.invoke(params))
            ##results.append(granger_causality_tool(params))
            results.append(granger_causality_tool.invoke(params))

            #print("Causation results:", results)
            
            return "\n".join(results)
            
        except Exception as e:
            print(f"Error analyzing causation: {e}")
            return None