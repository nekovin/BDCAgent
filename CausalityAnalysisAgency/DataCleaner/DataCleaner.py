from agency_swarm.agents import Agent

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.DataCleaningTool import DataCleaningTool

class DataCleaner(Agent):
    def __init__(self):
        super().__init__(
            name="DataCleaner",
            description="The DataCleaner agent is responsible for receiving CSV files, performing data cleaning, and preparing the data for analysis. It handles missing values, outliers, and other data quality issues.",
            instructions="./instructions.md",
            files_folder="./files",
            schemas_folder="./schemas",
            tools=[DataCleaningTool],
            tools_folder="./tools",
            temperature=0.3,
            max_prompt_tokens=25000,
        )

    def response_validator(self, message):
        return message
