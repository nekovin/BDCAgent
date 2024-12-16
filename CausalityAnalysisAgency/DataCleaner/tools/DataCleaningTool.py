from agency_swarm.tools import BaseTool
from pydantic import Field
import pandas as pd
from typing import Any, Optional, Dict, List

class DataCleaningTool(BaseTool):
    """
    This tool applies data cleaning techniques such as imputation or removal of missing values.
    """
    file_path: str = Field(
        ..., description="Path to the CSV file to be cleaned."
    )
    strategy: str = Field(
        ..., description="The strategy for handling missing values: 'mean', 'median', 'mode', 'drop_rows', or 'drop_columns'."
    )
    axis: Optional[int] = Field(
        0, description="Axis to drop rows or columns with missing values. 0 for rows, 1 for columns."
    )

    def run(self) -> Dict[str, List]:
        """
        Cleans the data according to the specified strategy.
        Returns data as a dictionary with column names and values.
        """
        try:
            data_frame = pd.read_csv(self.file_path)
            
            if self.strategy == 'mean':
                cleaned_data = data_frame.fillna(data_frame.mean())
            elif self.strategy == 'median':
                cleaned_data = data_frame.fillna(data_frame.median())
            elif self.strategy == 'mode':
                cleaned_data = data_frame.fillna(data_frame.mode().iloc[0])
            elif self.strategy == 'drop_rows':
                cleaned_data = data_frame.dropna(axis=self.axis)
            elif self.strategy == 'drop_columns':
                cleaned_data = data_frame.dropna(axis=self.axis)
            else:
                return {"error": ["Invalid strategy specified."]}

            # Convert DataFrame to dictionary
            return cleaned_data.to_dict(orient='list')
            
        except Exception as e:
            return {"error": [str(e)]}