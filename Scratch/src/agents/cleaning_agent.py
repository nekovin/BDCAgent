from langchain_openai import ChatOpenAI
from typing import List, Dict
import pandas as pd
import os
from langchain.tools import tool



@tool
def clean_dates_tool(data: Dict) -> Dict:
    """Converts date columns to datetime format"""
    return "data cleaned"

class CleaningAgent:
    def __init__(self, agent: ChatOpenAI):
        self.agent = agent
        
    def clean_data(self, data : Dict, formatted_instruction: str):
        try:
            if not os.path.exists("cleaned_data"):
                os.makedirs("cleaned_data")

            results = []
            cleaned_response = clean_dates_tool.invoke({'data' : data})

            try:
                print("Writing code")
                #print(formatted_instruction)
                formatted_instruction = "Please clean the data"
                #write_code = self.write_function(formatted_instruction, data)
                #print(write_code)
            except:
                print("Could not write code")
                pass

            results.append(cleaned_response)

            # save cleaned data' 
            cleaned_data = pd.DataFrame(data)
            cleaned_data.to_csv("cleaned_data/cleaned_data.csv", index=False)

            return "\n".join(results)
            
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return None
    
    def write_function(self, instruction: str, data: Dict):
            """
            Uses the ChatOpenAI agent to dynamically generate Python code based on the instruction.
            Args:
                instruction (str): Natural language instruction to generate the function code.
                data (Dict): Example data to provide context for the function.
            Returns:
                str: The generated Python code as a string.
            """
            try:
                prompt = f"""
                You are a coding assistant. Write a Python function based on the following instruction:

                Instruction:
                {instruction}

                The function should take a dictionary as input and output the transformed dictionary. Use the example data provided below as context for column names and types:

                Example data:
                {data}

                Ensure the code is well-commented and adheres to Python standards.
                """
                response = self.agent.invoke(prompt)
                generated_code = response['text'] 

                return generated_code
            except Exception as e:
                return f"Error in generating function code: {e}"