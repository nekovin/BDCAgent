from typing import Dict, Any
from models.models import Task, TaskType

from managers.task_manager import TaskManager

from agents.planning_agent import PlanningAgent
from agents.cleaning_agent import CleaningAgent
from agents.causation_agent import CausationAgent
from agents.interpretation_agent import InterpretationAgent

import pandas as pd

class AgentManager:
    def __init__(self, llm):
        self.llm = llm
        self.planning_agent = PlanningAgent(agent=llm)
        self.cleaning_agent = CleaningAgent(agent=llm)
        self.causation_agent = CausationAgent(agent=llm)
        self.interpretation_agent = InterpretationAgent(agent=llm)
        
        self.agent_calls = {
            TaskType.CLEAN: self.cleaning_agent.clean_data,
            TaskType.CORRELATION: self.causation_agent.analyze_causation,
            TaskType.INTERPRET: self.interpretation_agent.interpret_results
        }
    
    def execute_task(self, task: Task, df: pd.DataFrame, previous_result: Any = None) -> Any:
        message_history, response = self.planning_agent.send_message(task.raw_message)
        
        if task.type == TaskType.INTERPRET:
            return self.agent_calls[task.type](df, previous_result)
            
        if "Action:" in response:
            return self.agent_calls[task.type](response)
            
        return None
    
class Orchestrator:
    def __init__(self, llm):
        self.task_manager = TaskManager()
        self.agent_manager = AgentManager(llm)
        self.llm = llm
        
    def process(self, message: str, df : pd.DataFrame) -> str:
        tasks = self.task_manager.extract_tasks(self.llm, message)
    
        result = None
        
        for task in tasks:
            result = self.agent_manager.execute_task(task, df, result)
            if task.type == TaskType.INTERPRET:
                return f"Interpretation: \n\n{result}"
                
        return f"Could not interpret: \n\n{result}"