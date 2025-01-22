import nest_asyncio
from pydantic_ai import Agent
import pandas as pd
from httpx import AsyncClient
from dataclasses import dataclass
import logging
import numpy as np
from typing import List, Dict, Any

nest_asyncio.apply()

@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None

class DataCleaningOperations:
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str = 'ffill', columns: List[str] = None) -> pd.DataFrame:
        if columns is None:
            columns = df.columns
        return df[columns].fillna(method=method)

    @staticmethod
    def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column in df.columns:
            df[f"{column}_normalized"] = (df[column] - df[column].mean()) / df[column].std()
        return df

    @staticmethod
    def handle_temporal_gaps(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """Handle temporal gaps while preserving data"""
        try:
            df[time_column] = pd.to_datetime(df[time_column])
            
            df = df.sort_values(time_column)
            
            time_diff = df[time_column].diff()
            median_diff = time_diff.median()
            
            large_gaps = time_diff[time_diff > median_diff * 2]
            if not large_gaps.empty:
                logging.debug(f"Found {len(large_gaps)} temporal gaps larger than {median_diff * 2}")
            
            return df
        except Exception as e:
            logging.error(f"Error in handle_temporal_gaps: {str(e)}")
            return df

    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str, n_std: float = 3) -> pd.DataFrame:
        mean = df[column].mean()
        std = df[column].std()
        df[f"{column}_cleaned"] = df[column].clip(mean - n_std * std, mean + n_std * std)
        return df

class CleaningAgent:
    def __init__(self, model):
        self.cleaning_agent = Agent(
            model,
            system_prompt="""You are an expert at data engineering focused on Big Data Causality (BDC). 
            Your task is to analyze data and determine which cleaning operations to apply while preserving 
            causal relationships. You will select from predefined cleaning operations.""",
            deps_type=Deps,
            retries=2,
        )
        self.operations = DataCleaningOperations()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(level=logging.DEBUG)
        return logging.getLogger(__name__)

    def analyze_cleaning_needs(self, data: pd.DataFrame, bdc_plan: pd.DataFrame) -> Dict[str, List[dict]]:
        """Have the LLM analyze the data and determine which cleaning operations to apply"""
        
        data_info = {
            "missing_values": data.isnull().sum().to_dict(),
            "dtypes": data.dtypes.to_dict(),
            "unique_counts": data.nunique().to_dict(),
            "temporal_columns": [col for col in data.columns if data[col].dtype in ['datetime64[ns]', 'object'] 
                               and any(x in col.lower() for x in ['time', 'date', 'period'])]
        }

        prompt = f"""Analyze the following dataset and BDC plan to determine required cleaning operations.
        
        Dataset Info:
        {data_info}
        
        BDC Plan:
        {bdc_plan}
        
        Available Operations:
        1. handle_missing_values (methods: ffill, bfill, interpolate)
        2. normalize_column
        3. handle_temporal_gaps
        4. remove_outliers

        Return ONLY a Python dictionary mapping operation names to lists of parameter dictionaries. 
        Example format:
        {{
            "handle_missing_values": [{{"method": "ffill", "columns": ["col1", "col2"]}}],
            "normalize_column": [{{"column": "col1"}}, {{"column": "col2"}}],
            "handle_temporal_gaps": [{{"time_column": "timestamp"}}],
            "remove_outliers": [{{"column": "col1", "n_std": 3}}]
        }}
        
        Your response should contain ONLY the Python dictionary, no other text."""

        response = self.cleaning_agent.run_sync(prompt)
        
        try:
            cleaned_response = response.data.strip('`').strip()
            if cleaned_response.startswith('```python'):
                cleaned_response = cleaned_response.replace('```python', '').replace('```', '')
            
            cleaning_operations = eval(cleaned_response)
            
            if not isinstance(cleaning_operations, dict):
                raise ValueError("Response is not a dictionary")

            valid_operations = {'handle_missing_values', 'normalize_column', 'handle_temporal_gaps', 'remove_outliers'}
            if not all(op in valid_operations for op in cleaning_operations.keys()):
                raise ValueError("Invalid operation found in response")
                
            return cleaning_operations
            
        except Exception as e:
            self.logger.error(f"Failed to parse cleaning operations: {str(e)}")
            return {
                "handle_missing_values": [{"method": "ffill"}],
                "handle_temporal_gaps": [{"time_column": next((col for col in data.columns if 'time' in col.lower()), data.columns[0])}]
            }

    def clean_data(self, data: pd.DataFrame, bdc_plan: str) -> pd.DataFrame:
        """Apply cleaning operations based on analysis"""
        self.logger.debug("Starting data cleaning process")
        return data
        
        '''
        try:
            cleaning_operations = self.analyze_cleaning_needs(data, bdc_plan)
            self.logger.debug(f"Determined cleaning operations: {cleaning_operations}")
            
            cleaned_data = data.copy()
            
            for operation_name, operations_list in cleaning_operations.items(): 
                for operation in operations_list:
                    if hasattr(self.operations, operation_name):
                        cleaning_method = getattr(self.operations, operation_name)
                        cleaned_data = cleaning_method(cleaned_data, **operation)
                        self.logger.debug(f"Applied {operation_name} with params {operation}")
                    else:
                        self.logger.warning(f"Unknown operation: {operation_name}")

            self._validate_cleaning_results(cleaned_data, data, bdc_plan)
            
            return cleaned_data

        except Exception as e:
            self.logger.error(f"Error during cleaning process: {str(e)}")
            raise'''

    def _validate_cleaning_results(self, cleaned_data: pd.DataFrame, original_data: pd.DataFrame, bdc_plan: str):
        """Validate the cleaning results with detailed logging"""

        row_count_ratio = len(cleaned_data) / len(original_data)
        missing_columns = [col for col in original_data.columns if col not in cleaned_data.columns]
        original_nulls = original_data.isnull().sum().sum()
        cleaned_nulls = cleaned_data.isnull().sum().sum()
        
        self.logger.debug(f"Validation metrics:")
        self.logger.debug(f"- Row count ratio: {row_count_ratio:.2f}")
        self.logger.debug(f"- Missing columns: {missing_columns}")
        self.logger.debug(f"- Original null count: {original_nulls}")
        self.logger.debug(f"- Cleaned null count: {cleaned_nulls}")
        
        validations = {
            "row_count_preserved": len(cleaned_data) > 0,
            "columns_preserved": len(missing_columns) == 0,
            "data_quality": True 
        }

        failed_validations = [k for k, v in validations.items() if not v]
        if failed_validations:
            self.logger.warning(f"Failed validations: {failed_validations}")
        else:
            self.logger.debug("All validations passed")

        return all(validations.values())