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

        if data is None:
            response =  self.planning_agent.run_sync(f"Resond conversationally, inform the user that you can help with causal analysis: User response: {user_response}")
            return response.data, None

        # Request a string response from the planning agent
        raw_response = self.planning_agent.run_sync(
            f"""Context: {context}
                User response: {user_response}
                Data: {data}

                Task:
                1. Analyse the user's input and determine if it requires a plan or a simple conversational response.
                2. If a detailed plan is required or the user is asking for some kind of analysis, respond with "PLAN".
                3. If a conversational response is sufficient, respond with "RESPONSE: <your message here>"."""
        )

        print(f"DEBUG: Raw response: {raw_response.data}")
        response = raw_response.data.strip()
        plan = None

        # Check for the "PLAN" or "RESPONSE" keyword
        if response == "PLAN":
            print("Planning...")
            plan = self.plan_steps(data)
            response = f"Plans generated successfully. {plan}"
        elif response.startswith("RESPONSE:"):
            print("No planning...")
            response = response[len("RESPONSE:"):].strip()
        else:
            print("Unrecognized response format.")
            response = "Error: Unrecognized response format from planning agent."

        # Update memory with the response and any generated plans
        self.update_memory(user_response, data, plan)
        return response, plan

    
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

            The data has been cleaned and is ready for causal analysis.

            Return a step by step plan which will be fed into the causal agent for further analysis.

            Return the causal plan for the causal agent. """
                )
    
        return {'cleaning_plan' : cleaning_response.data, 'causal_plan':causal_response.data}
    
    def infer_response_old(self, user_response, data):
        context = "\n".join([f"Interaction {i+1}: {entry}" for i, entry in enumerate(self.memory)])

        if data is None:
            return  self.planning_agent.run_sync("Data is missing. Please provide data for analysis."), None
        
        raw_response = self.planning_agent.run_sync(
            f"""Context: {context}
                User response: {user_response}
                Data: {data}

                Task:
                1. Analyse the user's input and determine if it requires a detailed plan (e.g., task breakdown or execution steps) or a simple conversational response.
                2. If the input is clear and specifies a task, return:
                {{
                    "action": "plan"
                }}
                3. If the input is ambiguous (e.g., "any" or "you decide"), do the following:
                - Use the data provided to propose a starting point for the analysis or plan.
                - Generate a default exploratory plan based on available data.
                - Respond with this exploratory plan and state that it is based on inferred intent.
                4. If the input cannot be understood or there is insufficient data, return:
                {{
                    "action": "response",
                    "content": "Unable to proceed. Please provide more specific instructions."
                }}
                5. Always return a valid JSON object."""
        )

        print(type(raw_response))
        
        parsed_response = json.loads(raw_response.data)
        response = ""
        plan = ""
        #plan = self.plan_steps(data)
        #response = f"Plans generated successfully. {plan}"

        if parsed_response["action"] == "plan":
            print("Planning...")
            plan = self.plan_steps(data)
            response = f"Plans generated successfully. {plan}"
        else:
            print("No planning...")
            plan = ""
            response = parsed_response['content']
        
        self.update_memory(user_response, data, plan)

        return response, plan