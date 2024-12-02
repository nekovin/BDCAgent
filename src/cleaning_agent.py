from langchain_openai import ChatOpenAI
from typing import List, Dict
import pandas as pd
import os
from langchain.tools import tool

@tool
def clean_dates_tool(file_path: str) -> str:
    """Converts date columns to datetime format"""
    try:
        df = pd.read_csv(file_path)
        date_columns = df.columns[df.columns.str.contains('date', case=False)]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        df.to_csv("cleaned_data/cleaned_data.csv", index=False)
        return f"Dates cleaned in columns: {list(date_columns)}"
    except Exception as e:
        return f"Error cleaning dates: {str(e)}"

@tool
def remove_nulls_tool(file_path: str) -> str:
    """Removes rows with null values"""
    try:
        df = pd.read_csv(file_path)
        initial_rows = len(df)
        df = df.dropna()
        df.to_csv("cleaned_data/cleaned_data.csv", index=False)
        return f"Removed {initial_rows - len(df)} rows with null values"
    except Exception as e:
        return f"Error removing nulls: {str(e)}"

@tool
def fix_dtypes_tool(file_path: str) -> str:
    """Converts columns to appropriate data types"""
    try:
        df = pd.read_csv(file_path)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except:
                    continue
        df.to_csv('cleaned_data/cleaned_data.csv', index=False)
        return f"Fixed data types for numeric columns"
    except Exception as e:
        return f"Error fixing data types: {str(e)}"

class CleaningAgent:
    def __init__(self, agent: ChatOpenAI):
        self.agent = agent
        
    def clean_data(self, formatted_instruction: str):
        try:
            # Extract the file path
            file_path = formatted_instruction.split("Action Input:")[1].strip()
            
            if not os.path.exists("cleaned_data"):
                os.mkdir("cleaned_data")

            # Run all cleaning tools in sequence
            results = []
            #results.append(clean_dates_tool(file_path))
            results.append(clean_dates_tool.invoke(file_path))

            # after first cleaning
            file_path = r"cleanleaned_data/cleaned_data.csv"
            #results.append(remove_nulls_tool(file_path))
            results.append(remove_nulls_tool.invoke(file_path))
            #results.append(fix_dtypes_tool(file_path))
            results.append(fix_dtypes_tool.invoke(file_path))
            
            return "\n".join(results)
            
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return None