from agency_swarm.tools import BaseTool
from pydantic import Field
import pandas as pd
from typing import Any, Dict

class DataConsistencyTool(BaseTool):
    """
    This tool ensures data consistency and integrity by checking for and correcting inconsistencies
    in data types, formats, and values, ensuring that the dataset is uniform and ready for analysis.
    """

    data_frame: pd.DataFrame = Field(
        ..., description="The pandas DataFrame containing the data to be checked for consistency."
    )
    expected_types: Dict[str, str] = Field(
        ..., description="A dictionary specifying the expected data types for each column."
    )

    def run(self) -> Any:
        """
        Ensures data consistency by checking and correcting data types and formats.
        Returns the DataFrame with consistent data types and formats.
        """
        try:
            # Convert columns to the expected data types
            for column, expected_type in self.expected_types.items():
                if column in self.data_frame.columns:
                    self.data_frame[column] = self.data_frame[column].astype(expected_type)
            
            # Additional consistency checks can be added here, such as format corrections

            return self.data_frame
        except Exception as e:
            return f"An error occurred during data consistency checks: {e}"