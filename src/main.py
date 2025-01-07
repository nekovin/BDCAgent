import pandas as pd
from pydantic_ai.models.openai import OpenAIModel

from cleaning_agent.cleaning_agent import CleaningAgent
from causation_agent.causation_agent import CausationAgent
from planning_agent.planning_agent import PlanningAgent

from util.graph_utils import save_causal_graph

def get_agents(model):
    
    cleaning_agent = CleaningAgent(model)

    causation_agent = CausationAgent(model)

    return cleaning_agent, causation_agent

def convert_excel_to_csv(excel_path, csv_path):
    df = pd.read_excel(excel_path)
    df.to_csv(csv_path, index=False)
    print(f"Excel file converted to CSV and saved at: {csv_path}")

def get_data(file_path):

    df = pd.read_csv(file_path)

    sample_data = df.head(3)

    return sample_data

def get_args():
    #model = OpenAIModel("gpt-4o")

    model = OpenAIModel("gpt-4o-mini")
    convert_excel_to_csv(r"C:\Users\CL-11\OneDrive\Repos\BDCAgent\data\agric2A_72.xlsx", r"C:\Users\CL-11\OneDrive\Repos\BDCAgent\data\agric2A_72.csv")
    file_path = r"C:\Users\CL-11\OneDrive\Repos\BDCAgent\data\agric2A_72.csv"

    return model, file_path

def main():

    model, file_path = get_args()

    cleaning_agent, causation_agent = get_agents(model)

    df = get_data(file_path)

    # Planning
    planning_agent = PlanningAgent(model)

    while True:
            
        cleaning_plan, causal_plan = planning_agent.plan_steps(df)
        print("Cleaning Plan:\n", cleaning_plan)
        print("Causal Plan:\n", causal_plan)

        user_input = input("Do you want to proceed with the plan? (yes/no): ")
        if user_input.lower() == "yes":
            break

    cleaned_df = cleaning_agent.clean_data(df, cleaning_plan)
    cleaned_df.to_csv(r"C:\Users\CL-11\OneDrive\Repos\BDCAgent\data\cleaned_data.csv", index=False)
    print("Cleaned DataFrame:\n", cleaned_df)

    causal_df, causal_graph = causation_agent.analyse_causation(cleaned_df, causal_plan)
    causal_df.to_csv(r"C:\Users\CL-11\OneDrive\Repos\BDCAgent\data\causal_df.csv", index=False)
    print("Causal DataFrame:\n", causal_df)

    base_path = r"C:\Users\CL-11\OneDrive\Repos\BDCAgent\data\causal_graph"
    save_causal_graph(causal_graph, base_path)

    #visualize_causal_graph(cleaned_df, causation_agent)

    # add functionality to chat
    # maintain memory of previous chat
    # dyanmically update the plan based on the chat and recursively update if prompted

if __name__ == '__main__':
    main()