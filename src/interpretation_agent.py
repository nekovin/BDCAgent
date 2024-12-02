from langchain.tools import tool
from typing import List, Dict
from langchain.tools import tool
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import io
import sys
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import os

@dataclass
class InterpretationResult:
    summary: str
    insights: list[str]
    recommendations: list[str]
    confidence_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

class InterpretationAgent:
    def __init__(self, agent: Any, confidence_threshold: float = 0.7):
        """
        Initialize the InterpretationAgent with an LLM agent and confidence threshold.
        
        Args:
            agent: The language model agent to use for interpretation
            confidence_threshold: Minimum confidence score to consider an interpretation valid
        """
        self.agent = agent
        self.confidence_threshold = confidence_threshold
        self.interpretation_history = []
    
    def interpret(self, prompt: str) -> str:
        """
        Get raw interpretation from the agent.
        
        Args:
            prompt: The prompt to interpret
            
        Returns:
            Raw interpretation response
        """
        try:
            return self.agent.predict(prompt)
        except Exception as e:
            raise Exception(f"Error during interpretation: {e}")
    
    def interpret_results(self, formatted_instruction: str):
        try:
            # Parse correlation and Granger results
            correlation_line = formatted_instruction.split('\n')[0]
            correlation_value = float(correlation_line.split(': ')[1])
            
            # Create interpretation
            interpretation = []
            
            # Interpret correlation
            interpretation.append(f"The correlation coefficient of {correlation_value:.4f} indicates:")
            if abs(correlation_value) > 0.8:
                interpretation.append("- A very strong relationship between Open and Close prices")
            elif abs(correlation_value) > 0.6:
                interpretation.append("- A strong relationship between Open and Close prices")
            elif abs(correlation_value) > 0.4:
                interpretation.append("- A moderate relationship between Open and Close prices")
            else:
                interpretation.append("- A weak relationship between Open and Close prices")
                
            # Interpret Granger causality
            if "Granger" in formatted_instruction:
                lag_results = formatted_instruction.split('Lag')
                significant_lags = []
                
                for lag_result in lag_results[1:]:  # Skip first split which is correlation
                    if 'p=0.0000' in lag_result or 'p=0.00' in lag_result:
                        lag_num = lag_result.split(':')[0].strip()
                        significant_lags.append(lag_num)
                
                if significant_lags:
                    interpretation.append(f"\nGranger causality tests show significant relationships at lags: {', '.join(significant_lags)}")
                    interpretation.append("This suggests that past Open prices help predict Close prices")
            
            return "\n".join(interpretation)
            
        except Exception as e:
            print(f"Error analyzing interpretation: {e}")
            return None
    
    def _extract_summary(self, raw_interpretation: str) -> str:
        """Extract key summary from raw interpretation."""
        # Implementation would depend on the expected format of raw_interpretation
        # This is a placeholder implementation
        return raw_interpretation.split('\n')[0] if raw_interpretation else ""
    
    def _extract_insights(self, raw_interpretation: str) -> list[str]:
        """Extract key insights from raw interpretation."""
        # Placeholder implementation
        insights = []
        lines = raw_interpretation.split('\n')
        for line in lines:
            if line.startswith('- '):
                insights.append(line[2:])
        return insights
    
    def _extract_recommendations(self, raw_interpretation: str) -> list[str]:
        """Extract actionable recommendations from raw interpretation."""
        # Placeholder implementation
        recommendations = []
        lines = raw_interpretation.split('\n')
        in_recommendations = False
        for line in lines:
            if 'recommendations:' in line.lower():
                in_recommendations = True
                continue
            if in_recommendations and line.strip():
                recommendations.append(line.strip())
        return recommendations
    
    def _calculate_confidence(self, 
                            interpretation: str, 
                            data: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None
                            ) -> float:
        """
        Calculate confidence score for the interpretation.
        
        This could be based on factors like:
        - Consistency with historical interpretations
        - Data quality metrics
        - LLM confidence scores
        - Presence of key expected elements
        """
        # Placeholder implementation
        base_score = 0.8  # Default confidence
        
        # Adjust based on interpretation length
        if len(interpretation) < 50:
            base_score -= 0.2
            
        # Adjust based on data presence
        if data is not None:
            base_score += 0.1
            
        return min(1.0, max(0.0, base_score))
    
    def _validate_interpretation(self, result: InterpretationResult) -> bool:
        """Validate interpretation result meets quality standards."""
        if result.confidence_score < self.confidence_threshold:
            return False
        if not result.summary or not result.insights:
            return False
        return True
    
    def _get_data_shape(self, data: Optional[Union[pd.DataFrame, Dict[str, Any]]]) -> Optional[tuple]:
        """Get shape of input data if available."""
        if isinstance(data, pd.DataFrame):
            return data.shape
        elif isinstance(data, dict):
            return (len(data),)
        return None
    
    def get_interpretation_history(self) -> list[InterpretationResult]:
        """Get history of all successful interpretations."""
        return self.interpretation_history