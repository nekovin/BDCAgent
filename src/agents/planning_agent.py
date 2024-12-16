from typing import Dict, Any, List
from models.models import Task, TaskType
from agents.cleaning_agent import CleaningAgent
from agents.causation_agent import CausationAgent
from agents.interpretation_agent import InterpretationAgent
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_openai import ChatOpenAI
import pandas as pd

class PlanningAgent:   
    def __init__(self, agent: ChatOpenAI, message_history=None):
        self.agent = agent
        self.message_history = message_history or FileChatMessageHistory("config/chat_history.json")
        self.cleaning_agent = CleaningAgent(agent=agent)
        self.causation_agent = CausationAgent(agent=agent) 
        self.interpretation_agent = InterpretationAgent(agent=agent)
        
        self.agent_calls = {
            TaskType.CLEAN: self.cleaning_agent.clean_data,
            TaskType.CORRELATION: self.causation_agent.analyze_causation,
            TaskType.INTERPRET: self.interpretation_agent.interpret_results
        }

        self.TASK_PROMPT = """
        Please format it like this: '<Task 1>' \n '<Task 2>' \n '<Task 3>', without any extra information such as numbers or bullet points.
        The available tools are CleanData, FindCorrelation
        If FindCorrelation, please specify the variables to find the correlation between. For example, 'FindCorrelation [var1, var2]'.
        """

    def execute_task(self, message_history, task: Task, data: Dict, previous_result: Any = None) -> Any:
        message_history, response = self.send_message(task.raw_message, messages_history=message_history)

        if task.type == TaskType.INTERPRET:
            return self.agent_calls[task.type](data, previous_result), message_history
            
        elif "Action:" in response:
            return self.agent_calls[task.type](data, response), message_history
            
        return None

    def process(self, message: str, data: Dict) -> str:
        result = None
        tasks = self.extract_tasks(self.agent, message)

        for task in tasks:
            if task.type == TaskType.INTERPRET:
                return f"Interpretation: \n\n{result}", self.message_history
            
            result, message_history = self.execute_task(self.message_history, task, data, result)

            self.message_history.add_user_message(task.raw_message)
            self.message_history.add_ai_message(str(result))

            if task.type == TaskType.CLEAN:
                try:
                    print("Loading cleaned data")
                    data = pd.read_csv("cleaned_data/cleaned_data.csv").to_dict()
                except:
                    print("Could not load cleaned data")
                    pass
                
        return f"Could not interpret: \n\n{result}", self.message_history

    def predict(self, prompt: str):
        return self.agent.invoke(prompt)

    def send_message_block(self, new_message_block: Dict, messages_history: List[Dict] = None):
        user_input = new_message_block["content"][0]["text"]
        
        try:
            format_prompt = """Format this request into the following structure exactly:
            If it's a cleaning request, use this format ONLY:
            I need to clean the data
            Action: CleanData

            If it's a correlation request, use this format ONLY:
            I need to find the correlation
            Action: FindCorrelation

            Format ONLY ONE of these based on the request.
            Request: """

            formatted_response = self.predict(format_prompt + user_input).content
                        
            return messages_history, formatted_response
            
        except Exception as e:
            print(f"Error: {e}")
            return messages_history, formatted_response

    def send_message(self, new_message: str, messages_history: List[Dict] = None):
        if messages_history is None:
            messages_history = []
            
        new_message_block = {
            "role": "user",
            "content": [{"type": "text", "text": new_message}]
        }

        return self.send_message_block(new_message_block=new_message_block, messages_history=messages_history)
    
    def extract_tasks(self, llm, message: str) -> List[Task]:
        message = llm.invoke(f"Extract tasks from the following message:\n\n{message}.").content
        response = llm.invoke([
            {"role": "system", "content": self.TASK_PROMPT}, 
            {"role": "user", "content": message}
        ])
        
        tasks = []
        for task_str in response.content.splitlines():
            if task_str:
                task_type = self._determine_task_type(task_str)
                variables = self._extract_variables(task_str) if task_type == TaskType.CORRELATION else None
                tasks.append(Task(type=task_type, raw_message=task_str, variables=variables))
                
        tasks.append(Task(type=TaskType.INTERPRET, raw_message="Interpret"))
        return tasks
    
    def _determine_task_type(self, task_str: str) -> TaskType:
        task_str = task_str.lower()
        if "clean" in task_str:
            return TaskType.CLEAN
        elif "correlation" in task_str:
            return TaskType.CORRELATION
        return TaskType.INTERPRET
    
    def _extract_variables(self, task_str: str) -> List[str]:
        if '[' in task_str and ']' in task_str:
            vars_str = task_str.split('[')[1].split(']')[0]
            return [v.strip() for v in vars_str.split(',')]
        return []