from agency_swarm.tools import BaseTool
from pydantic import Field
from typing import Any, Dict

class GenerateUserFriendlyReportTool(BaseTool):
    """
    This tool generates user-friendly reports or presentations based on the interpreted results.
    It formats the insights and explanations in a clear and accessible manner for users.
    """

    interpreted_results: Dict[str, Any] = Field(
        ..., description="The interpreted results containing insights and explanations to be included in the report."
    )

    def run(self) -> str:
        """
        Generates a user-friendly report based on the interpreted results.
        Returns the formatted report as a string.
        """
        try:
            # Extract necessary information from the interpreted results
            method = self.interpreted_results.get("method", "Unknown Method")
            score = self.interpreted_results.get("score", "N/A")
            summary = self.interpreted_results.get("summary", "No summary available.")

            # Format the report
            report = (
                f"--- Causality Analysis Report ---\n\n"
                f"Analysis Method: {method}\n"
                f"Performance Score (R^2): {score}\n\n"
                f"Summary of Findings:\n{summary}\n\n"
                f"--- End of Report ---"
            )

            return report
        except Exception as e:
            return f"An error occurred while generating the report: {e}"