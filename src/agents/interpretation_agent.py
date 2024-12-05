from langchain.tools import tool
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import json

@dataclass
class InterpretationResult:
    summary: str
    insights: List[str]
    recommendations: List[str]
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
    
    def interpret_results(self, data : pd.DataFrame, prompt: str) -> str:
        """
        Get raw interpretation from the agent.
        
        Args:
            prompt: The prompt to interpret
            
        Returns:
            Raw interpretation response
        """
        try:

            # Prepare formatted input
            formatted_input = f"Data:\n{data.head().to_string(index=False)}\n\nPrompt:\n{prompt}"
            #if isinstance(formatted_input, dict):
            #    formatted_input = json.dumps(formatted_input)  

            context = "Interpret the data provided above. Please be descriptive. Format with ### above and below."
            response = self.agent.invoke([{"role": "system", "content": context}, {"role": "user", "content": formatted_input}])
            return response.content

        except Exception as e:
            raise Exception(f"Error during interpretation: {e}")