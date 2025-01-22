from pydantic_ai import Agent
from httpx import AsyncClient
from dataclasses import dataclass
import pandas as pd
import json

@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None

class PlanningAgent:
    def __init__(self, model):
        self.planning_agent = Agent(
            model,
            system_prompt="""You are an expert planning agent specializing in causal analysis. 
            Your primary focus is on understanding and mapping the causal relationships between variables over time in cleaning operations.
            You will be creating two different plans for the cleaning and causal agents to follow if prompted so.
            Be very concise with your responses""",
            deps_type=Deps,
            retries=2
        )

        self.memory = []

    def get_planning_agent(self):
        return self.planning_agent
    
    def update_memory(self, user_response, data, plan=None):
        self.memory.append({
            "user_response": user_response,
            "data": data,
            "plan": plan
        })
    
    def infer_response(self, user_response, data):

        context = "\n".join([f"Interaction {i+1}: {entry}" for i, entry in enumerate(self.memory)])

        # load input 
        # respond

        raw_response = self.planning_agent.run_sync(
            f"""Context: {context}
            User response: {user_response}
            Data: {data}

            Task:
            1. Determine if the user's input requires a plan or a conversational response.
            2. If a plan is required, return:
            {{
                "action": "plan",
            }}
            3. If no plan is required, return:
            {{
                "action": "response",
                "content": "Conversational response here..."
            }}
            4. Always return a valid JSON object."""
        )

        # Parse the JSON response
        #print(raw_response)

        parsed_response  = json.loads(raw_response.data)

        response = ""

        # does this indicate requirement for a plan 
        # if so, generate plan

        #if parsed_response["action"] == "plan":
            
        plan = self.plan_steps(data)
        response = f"Plans generated successfully.{plan}"

        #else:
            #plan = None
            #response = parsed_response['content']
            
        self.update_memory(user_response, data, plan)

        return response, plan

        '''
        if data is not None and not data.empty:
            cleaning_plan, causal_plan = self.plan_steps(data)
            response = "Plans generated successfully."
            return response, [cleaning_plan, causal_plan]
        else:
            response = self.planning_agent.run_sync("Data is empty or invalid. Please upload valid data.")
            return response, None'''
    
    def plan_steps(self, data):
        
        bdc_format = pd.DataFrame(
            {
            "Time": ["n", "n+1", "n+2", "n+..."],
            "Variable 1": ["", "", "", ""],
            "Variable 2": ["", "", "", ""],
            "...": ["", "", "", ""],
            "Variable N": ["", "", "", ""]
            }
        )

        cleaning_response = self.planning_agent.run_sync(
            f"""Analyze the following cleaning data: {data}
        Return a step by step plan so that the final result after the cleaning agent is in the following format:
        {bdc_format}

        The available operations are:
        1. handle_missing_values (methods: ffill, bfill, interpolate)
        2. normalize_column
        3. handle_temporal_gaps
        4. remove_outliers

        You must identify the time series data and the variables that need to be cleaned and formatted into the appropriate format.
        
        Return the cleaning plan for the cleaning agent."""
                )
        
        causal_response = self.planning_agent.run_sync(
            f"""Analyze the following data: {data}

            Return a step by step plan which will be fed into the causal agent for further analysis.

            Return the causal plan for the causal agent. """
                )
    
        return {'cleaning_plan' : cleaning_response.data, 'causal_plan':causal_response.data}