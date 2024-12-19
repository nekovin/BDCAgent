import nest_asyncio
from pydantic_ai import Agent

import pandas as pd
from httpx import AsyncClient
import re
from dataclasses import dataclass

nest_asyncio.apply()

@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None


class CausationAgent:
    def __init__(self, model):
        self.causation_agent = Agent(
            model,
            system_prompt=f'You are an expert in causal inference and causality analysis. '
                          f'Your task is to identify causal relationships, perform causal reasoning, and analyse data.',
            deps_type=Deps,
            retries=2
        )

    def get_causation_agent(self):
        return self.causation_agent

    async def identify_causes():
        pass

    def analyse_causation(self, data):

        # Perform causation analysis
        response = self.causation_agent.run_sync(
            f"Perform causal analysis on the following data and identify causal relationships: {data}"
        )
        print(response.data)

        # Ask the causation agent to generate a Python function
        response = self.causation_agent.run_sync(
            f"Based on the following data, create a Python function called `analyse_causation` "
            f"that takes a pandas DataFrame as input and identifies causal relationships in the data. "
            f"Return only the function ready to copy and paste directly without any extra text. "
            f"Data: {data}"
        )

        print(response.data)

        function_match = re.search(
            r'def\s+analyse_causation\s*\([^)]*\)\s*->\s*pd\.DataFrame\s*:\s*(.*?)\n\s*return\s+\w+',
            response.data,
            re.DOTALL | re.MULTILINE
        )

        if function_match:
            print("Function found:")
            print(function_match.group(0))
        else:
            print("No function found")

        namespace = {"pd": pd}
        exec(function_match.group(0), namespace)

        # Access the analyse_causation function dynamically
        dynamic_analyse_causation = namespace['analyse_causation']
        causal_df = dynamic_analyse_causation(data)

        print("Causal Analysis Results:\n", causal_df)
        return causal_df
