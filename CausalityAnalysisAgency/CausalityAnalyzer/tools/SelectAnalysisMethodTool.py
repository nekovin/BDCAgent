from agency_swarm.tools import BaseTool
from pydantic import Field
import pandas as pd
from typing import Any, List, Dict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class SelectAnalysisMethodTool(BaseTool):
    """
    This tool allows the CausalityAnalyzer agent to select and apply different statistical and machine learning methods
    for causality analysis. It provides options for various techniques and ensures the selected method is applied correctly
    to the data.
    """

    data_frame: pd.DataFrame = Field(
        ..., description="The pandas DataFrame containing the data for causality analysis."
    )
    target_variable: str = Field(
        ..., description="The target variable for the causality analysis."
    )
    feature_variables: List[str] = Field(
        ..., description="A list of feature variables to be used in the analysis."
    )
    method: str = Field(
        ..., description="The analysis method to apply. Options include 'linear_regression' and 'random_forest'."
    )

    def run(self) -> Any:
        """
        Selects and applies the specified analysis method to the data.
        Returns the model's performance score and a brief summary of the analysis.
        """
        try:
            if self.target_variable not in self.data_frame.columns:
                return f"Target variable '{self.target_variable}' not found in the data."

            for feature in self.feature_variables:
                if feature not in self.data_frame.columns:
                    return f"Feature variable '{feature}' not found in the data."

            X = self.data_frame[self.feature_variables].values
            y = self.data_frame[self.target_variable].values

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if self.method == 'linear_regression':
                model = LinearRegression()
            elif self.method == 'random_forest':
                model = RandomForestRegressor(random_state=42)
            else:
                return f"Method '{self.method}' is not supported. Please choose 'linear_regression' or 'random_forest'."

            # Fit the model and evaluate its performance
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = r2_score(y_test, predictions)

            return {
                "method": self.method,
                "score": score,
                "summary": f"The {self.method} model achieved an R^2 score of {score:.2f}."
            }
        except Exception as e:
            return f"An error occurred during analysis method selection and application: {e}"