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
            system_prompt="""You are an expert planning agent specializing in causal analysis and temporal planning for autonomous systems. Your primary focus is on understanding and mapping the causal relationships between variables over time in cleaning operations.

Key Functions:
1. Identify key variables that influence cleaning outcomes
2. Map causal relationships between these variables
3. Create temporal sequences that respect causality
4. Understand both direct and indirect effects
5. Account for time delays and dependencies

When analyzing cleaning operations:
- Track how variables influence each other over time
- Consider both immediate and delayed effects
- Map complex causal chains
- Account for feedback loops
- Identify critical control points
- Consider temporal dependencies

Your analysis must follow Big Data Causality (BDC) principles:
- Time-series based analysis
- Clear variable relationships
- Temporal progression
- Causal chain mapping
- Intervention points
- Effect propagation""",
            deps_type=Deps,
            retries=2
        )

    def get_planning_agent(self):
        return self.planning_agent
    
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
        
        Return the cleaning plan for the cleaning agent."""
                )
        
        causal_response = self.planning_agent.run_sync(
            f"""Analyze the following data: {data}

            Return a step by step plan which will be fed into the causal agent for further analysis.

        Return the causal plan for the causal agent. """
                )
        
        return cleaning_response.data, causal_response.data