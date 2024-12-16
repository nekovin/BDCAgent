from agency_swarm.tools import BaseTool
from pydantic import Field
from typing import Any, Dict

class ReceiveAnalysisResultsTool(BaseTool):
    """
    This tool enables the ResultInterpreter agent to receive analysis results from the CausalityAnalyzer agent.
    It handles data input and ensures the results are in a suitable format for interpretation.
    """

    analysis_results: Dict[str, Any] = Field(
        ..., description="The analysis results from the CausalityAnalyzer agent, including method, score, and summary."
    )

    def run(self) -> Any:
        """
        Processes the received analysis results to ensure they are in a suitable format for interpretation.
        Returns the results if they are valid and ready for interpretation.
        """
        try:
            # Validate the structure of the analysis results
            required_keys = {"method", "score", "summary"}
            if not required_keys.issubset(self.analysis_results.keys()):
                return "The analysis results are missing required keys."

            # Here, we simply return the analysis results assuming they are already in the correct format
            return self.analysis_results
        except Exception as e:
            return f"An error occurred while processing the analysis results: {e}"