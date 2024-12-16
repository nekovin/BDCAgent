from agency_swarm.tools import BaseTool
from pydantic import Field
import pandas as pd
from typing import Any

class ReceiveAndReadCSVTool(BaseTool):
    """
    This tool enables the DataCleaner agent to receive and read CSV files.
    It handles file input and parses the CSV content into a structured format for further processing.
    """

    file_path: str = Field(
        ..., description="The file path of the CSV file to be read."
    )

    def run(self) -> Any:
        """
        Reads the CSV file from the specified file path and returns its content as a pandas DataFrame.
        This structured format allows for further data processing and analysis.
        """
        try:
            # Read the CSV file into a pandas DataFrame
            data_frame = pd.read_csv(self.file_path)
            return data_frame
        except Exception as e:
            return f"An error occurred while reading the CSV file: {e}"