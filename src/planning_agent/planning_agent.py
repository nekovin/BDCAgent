from pydantic_ai import Agent
from httpx import AsyncClient
from dataclasses import dataclass
import pandas as pd

@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None

class PlanningAgent:
    def __init__(self, model):

        self.planning_agent = Agent(
            model,
            system_prompt=f'You are an expert at making plans for agents. Your task is to plan the steps for the agents.',
            deps_type=Deps,
            retries=2
        )

    def get_planning_agent(self):
        return self.cleaning_agent
    
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

        response = self.planning_agent.run_sync(
            f"Plan the steps for the agents: {data}. "
            f"The final format should be the following: {bdc_format}"
            f"Return the steps for the cleaning agent")
        
        return response.data