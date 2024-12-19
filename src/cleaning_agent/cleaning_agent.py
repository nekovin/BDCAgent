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


class CleaningAgent:
    def __init__(self, model):
        self.cleaning_agent = Agent(
            model,
            system_prompt="You are an expert at data engineering. Your task is to clean and preprocess data.",
            deps_type=Deps,
            retries=2,
        )

    def get_cleaning_agent(self):
        return self.cleaning_agent

    def clean_data(self, data, plan):
        structured_plan = "\n".join(
            [f"{step}: {instruction}" for step, instruction in enumerate(plan, 1)]
        )
        response = self.cleaning_agent.run_sync(
            f"Using the following cleaning plan, clean the provided dataset:\n\n"
            f"Cleaning Plan:\n{structured_plan}\n\n"
            f"Dataset: {data}"
        )

        attempt = 0

        while attempt < 3:
            response = self.cleaning_agent.run_sync(
                f"Create a Python function called `clean_data` "
                f"that takes a pandas DataFrame as input and applies the following plan:\n\n"
                f"{structured_plan}\n\n"
                f"Return only the function ready to copy and paste directly. "
                f"I will then use a function match to extract the function ready for execution: re.search(r'def\s+clean_data\s*\([^)]*\)\s*:\s*(.*?)\n\s*return\s+\w+',response.data,re.DOTALL | re.MULTILINE"
            )

            function_match = re.search(
                r'def\s+clean_data\s*\([^)]*\)\s*:\s*(.*?)\n\s*return\s+\w+',
                response.data,
                re.DOTALL | re.MULTILINE
            )

            if function_match:
                namespace = {"pd": pd}
                exec(function_match.group(0), namespace)

                dynamic_clean_data = namespace["clean_data"]

                cleaned_df = dynamic_clean_data(data)
                return cleaned_df

            attempt += 1

        return None
