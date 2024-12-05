
from planning_agent import *
from cleaning_agent import *
from causation_agent import *
from interpretation_agent import *

def extract_tasks(llm, message):
    context = "\n\n Please format it like this: '<Task 1>' \n '<Task 2>' \n '<Task 3>', ... without any extra information such as numbers or bullet points.\n\n The availabe tools are CleanData, FindCorrelation"
    context = context + "If FindCorrelation, please specify the variables to find the correlation between. For example, 'FindCorrelation [var1, var2]'."
    message = llm.invoke(f"Extract tasks from the following message:\n\n{message}.").content
    response = llm.invoke([{"role": "system", "content": context}, {"role": "user", "content": message}])
    response = response.content 

    tasks = response.splitlines()
    return tasks

def extract_agents(messages: List[str]) -> Dict[str, str]:
    agent_mapping = {
        "clean": "CleanData",
        "correlation": "FindCorrelation",
        "interpret": "Interpret"
    }
    
    needed_agents = {}
    for message in messages:
        message = message.lower()
        for keyword, agent in agent_mapping.items():
            if keyword in message:
                needed_agents[agent] = True
    
    return needed_agents

def prompt(llm, message, file_path):
    # define agents, planning agent should dynamically adjust which agentss are needed
    planning_agent = PlanningAgent(agent=llm)
    cleaning_agent = CleaningAgent(agent=llm)
    causation_agent = CausationAgent(agent=llm)
    interpretation_agent = InterpretationAgent(agent=llm)

    agent_calls = {
        "CleanData": cleaning_agent.clean_data,
        "FindCorrelation": causation_agent.analyze_causation,
        "Interpret": interpretation_agent.interpret_results
    }
    
    messages = extract_tasks(llm, message) + ["Interpret"]
    agents = extract_agents(messages)

    print("Processing")

    for message in messages:
        message_history, response = planning_agent.send_message(message)
        formatted_response = response.replace("[file_path]", file_path)

        if message == "Interpret":
            cleaned_data_path = "../cleaned_data/cleaned_data.csv" # this should be dynamic
            interpretation = agent_calls["Interpret"](cleaned_data_path, result)
            return "Interpretation: \n\n" + interpretation

        if "Action:" in formatted_response:

            action = formatted_response.split("Action:")[1].split("\n")[0].strip()
            if action in agent_calls:
                result = agent_calls[action](formatted_response)
        
    return "Could not interpret: \n\n" + result