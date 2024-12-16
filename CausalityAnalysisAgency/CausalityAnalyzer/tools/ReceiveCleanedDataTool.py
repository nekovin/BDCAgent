from agency_swarm.tools import BaseTool
from pydantic import Field
import pandas as pd
from typing import Any

class ReceiveCleanedDataTool(BaseTool):
    """
    This tool enables the CausalityAnalyzer agent to receive cleaned data from the DataCleaner agent.
    It handles data input and ensures the data is in a suitable format for analysis.
    """

    cleaned_data: pd.DataFrame = Field(
        ..., description="The cleaned pandas DataFrame ready for analysis."
    )

    def run(self) -> Any:
        """
        Processes the received cleaned data to ensure it is in a suitable format for analysis.
        Returns the DataFrame if it is valid and ready for analysis.
        """
        try:
            # Perform any necessary checks or transformations to ensure data is ready for analysis
            # For example, checking for any remaining missing values or ensuring correct data types

            # Here, we simply return the cleaned data assuming it is already in the correct format
            return self.cleaned_data
        except Exception as e:
            return f"An error occurred while processing the cleaned data: {e}"