from typing import List
from models.models import Task, TaskType

class TaskManager:
    def __init__(self):
        self.TASK_PROMPT = """
        Please format it like this: '<Task 1>' \n '<Task 2>' \n '<Task 3>', without any extra information such as numbers or bullet points.
        The available tools are CleanData, FindCorrelation
        If FindCorrelation, please specify the variables to find the correlation between. For example, 'FindCorrelation [var1, var2]'.
        """
    
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