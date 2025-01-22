import nest_asyncio
from pydantic_ai import Agent
import pandas as pd
import numpy as np
from httpx import AsyncClient
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import networkx as nx

nest_asyncio.apply()

@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None

class CausationAgent:
    def __init__(self, model):
        self.causation_agent = Agent(
            model,
            system_prompt="""You are an expert in causal inference and causality analysis. 
            Your task is to identify causal relationships and analyze data.""",
            deps_type=Deps,
            retries=2
        )

    def analyze_causation(self, data: pd.DataFrame, planning_data: str) -> Tuple[pd.DataFrame, nx.DiGraph]:
        """Analyze causal relationships in the data and return summary DataFrame and causal graph.
        
        Args:
            data: Input DataFrame containing the variables
            planning_data: String containing planning information
            
        Returns:
            Tuple containing:
            - DataFrame summarizing causal relationships
            - NetworkX DiGraph representing the causal network
        """
        # introduce planning data here
        response = self.causation_agent.run_sync(
            f"""Do some super basic causal analysis on the data please. Heres the plan from the planning agent: {planning_data} Data{data}"""
        )

        return response.data
    
        # Initialize results storage
        summary_data = []
        G = nx.DiGraph()

        # Analyze relationships between variables
        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                try:

                    correlation = data[var1].corr(data[var2])

                    print(correlation)

                    temporal_corr = data[var1].corr(data[var2].shift(-1))
                    reverse_temporal_corr = data[var2].corr(data[var1].shift(-1))
                    
                    # Determine causality direction and strength
                    if abs(temporal_corr) > abs(reverse_temporal_corr):
                        cause, effect = var1, var2
                        strength = abs(temporal_corr)
                    else:
                        cause, effect = var2, var1
                        strength = abs(reverse_temporal_corr)

                    # Only include relationships with meaningful strength
                    if strength > 0.3:  # Threshold for significance
                        summary_data.append({
                            'Cause': cause,
                            'Effect': effect,
                            'Correlation': correlation,
                            'Causal_Strength': strength
                        })
                        
                        # Add edge to graph
                        G.add_edge(cause, effect, weight=strength)
                        
                except Exception as e:
                    print(f"Error analyzing {var1} -> {var2}: {str(e)}")
                    continue

        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by causal strength
        if not summary_df.empty:
            summary_df = summary_df.sort_values('Causal_Strength', ascending=False)

        return summary_df, G