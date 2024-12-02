from langchain_openai import ChatOpenAI
from typing import List, Dict

class PlanningAgent:   
    def __init__(self, agent: ChatOpenAI):
        self.agent = agent

    def predict(self, prompt: str):
        #return self.agent.predict(prompt)
        return self.agent.invoke(prompt)

    def send_message_block(self, new_message_block: Dict, messages_history: List[Dict] = None):
        user_input = new_message_block["content"][0]["text"]
        
        try:
            format_prompt = """Format this request into the following structure exactly:
            If it's a cleaning request, use this format ONLY:
            I need to clean the data
            Action: CleanData
            Action Input: [file_path]

            If it's a correlation request, use this format ONLY:
            I need to find the correlation
            Action: FindCorrelation
            Action Input: [file_path] [column1,column2]

            Format ONLY ONE of these based on the request.
            Request: """
            
            #formatted_response = self.predict(format_prompt + user_input)
            formatted_response = self.predict(format_prompt + user_input).content

                        
            return messages_history, formatted_response
            
        except Exception as e:
            print(f"Error: {e}")
            return messages_history, formatted_response

    def send_message(self, new_message: str, messages_history: List[Dict] = None):
        """Send a new message to the agent."""
        if messages_history is None:
            messages_history = []
            
        new_message_block = {
            "role": "user",
            "content": [{"type": "text", "text": new_message}]
        }

        return self.send_message_block(new_message_block=new_message_block, messages_history=messages_history)