from agency_swarm.tools import BaseTool
from pydantic import Field
from typing import Any, Dict

class SummarizeCausalFindingsTool(BaseTool):
    """
    This tool summarizes the causal findings in a concise manner, highlighting key insights and implications.
    It distills complex results into essential points that are easy to understand.
    """

    causal_results: Dict[str, Any] = Field(
        ..., description="The causal analysis results containing detailed findings to be summarized."
    )

    def run(self) -> str:
        """
        Summarizes the causal findings into essential points.
        Returns the summary as a string.
        """
        try:
            # Extract necessary information from the causal results
            method = self.causal_results.get("method", "Unknown Method")
            score = self.causal_results.get("score", "N/A")
            summary = self.causal_results.get("summary", "No detailed findings available.")

            # Create a concise summary
            concise_summary = (
                f"Key Insights from Causal Analysis:\n"
                f"- Method Used: {method}\n"
                f"- Performance Score: {score}\n"
                f"- Summary: {summary}\n"
                f"Implications: The analysis suggests that the method used provides a reliable measure of causality, "
                f"with a performance score indicating the model's accuracy. Further investigation may be required to "
                f"explore additional variables or refine the model for better insights."
            )

            return concise_summary
        except Exception as e:
            return f"An error occurred while summarizing the causal findings: {e}"