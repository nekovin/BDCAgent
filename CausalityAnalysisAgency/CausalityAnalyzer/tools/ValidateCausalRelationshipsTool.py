from agency_swarm.tools import BaseTool
from pydantic import Field
import pandas as pd
from typing import Any, List, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

class ValidateCausalRelationshipsTool(BaseTool):
    """
    This tool validates identified causal relationships using cross-validation or other statistical methods.
    It ensures the robustness and reliability of the causal findings.
    """

    data_frame: pd.DataFrame = Field(
        ..., description="The pandas DataFrame containing the data with potential causal relationships."
    )
    causal_pairs: List[Tuple[str, str]] = Field(
        ..., description="A list of tuples representing the causal relationships to validate, where each tuple is (cause, effect)."
    )
    cv_folds: int = Field(
        5, description="The number of cross-validation folds to use for validation."
    )

    def run(self) -> Any:
        """
        Validates the identified causal relationships using cross-validation.
        Returns a dictionary with causal pairs as keys and their validation scores as values.
        """
        try:
            validation_results = {}

            for cause, effect in self.causal_pairs:
                if cause in self.data_frame.columns and effect in self.data_frame.columns:
                    X = self.data_frame[[cause]].values
                    y = self.data_frame[effect].values

                    # Use linear regression as a simple model for validation
                    model = LinearRegression()
                    scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='r2')

                    # Store the mean cross-validation score for the causal pair
                    validation_results[(cause, effect)] = np.mean(scores)

            return validation_results
        except Exception as e:
            return f"An error occurred during causal relationship validation: {e}"