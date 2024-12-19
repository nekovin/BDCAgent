import pandas as pd
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel

from cleaning_agent.cleaning_agent import CleaningAgent
from causation_agent.causation_agent import CausationAgent
from planning_agent.planning_agent import PlanningAgent

def get_agents(model):
    
    cleaning_agent = CleaningAgent(model)

    causation_agent = CausationAgent(model)

    return cleaning_agent, causation_agent

def get_data(file_path):
    df = pd.read_csv(file_path)

    sample_data = df.head(3)#.to_dict('records')

    return sample_data

def get_args():
    model = OpenAIModel("gpt-4o")
    file_path = r"C:\Users\CL-11\OneDrive\Repos\CausalAgent\data\AA.csv"

    return model, file_path

def main():

    model, file_path = get_args()

    cleaning_agent, causation_agent = get_agents(model)

    df = get_data(file_path)

    # Planning
    planning_agent = PlanningAgent(model)

    plan = planning_agent.plan_steps(df)

    print("Plan:\n", plan)

    # Cleaning
    cleaned_df = cleaning_agent.clean_data(df, plan)

    cols = [col for col in df.columns if "Variable" in col or "Time" in col]

    print(cols)

    final_df = cleaned_df[cols]

    cleaned_df.to_csv(r"C:\Users\CL-11\OneDrive\Repos\CausalAgent\data\cleaned_data.csv", index=False)
    final_df.to_csv(r"C:\Users\CL-11\OneDrive\Repos\CausalAgent\data\formatted_data.csv", index=False)
    print("Cleaned DataFrame:\n", cleaned_df)

    # Causation
    #causal_df = causation_agent.analyse_causation(cleaned_df)
    #causal_df.to_csv(r"C:\Users\CL-11\OneDrive\Repos\CausalAgent\data\causal_df.csv", index=False)
    #print("Causal DataFrame:\n", causal_df)


if __name__ == '__main__':
    main()