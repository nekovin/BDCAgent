from agency_swarm import Agency
from ResultInterpreter import ResultInterpreter
from CausalityAnalyzer import CausalityAnalyzer
from DataCleaner import DataCleaner
from CausalityCEO import CausalityCEO

import pandas as pd

causality_ceo = CausalityCEO()
data_cleaner = DataCleaner()
causality_analyzer = CausalityAnalyzer()
result_interpreter = ResultInterpreter()

agency = Agency([causality_ceo, [causality_ceo, data_cleaner],
                 [data_cleaner, causality_analyzer],
                 [causality_analyzer, result_interpreter],
                 [result_interpreter, causality_ceo]],
                shared_instructions='./agency_manifesto.md',  # shared instructions for all agents
                max_prompt_tokens=25000,  # default tokens in conversation for all agents
                temperature=0.3,  # default temperature for all agents
                )

def main():
    data_cleaner = DataCleaner()
    file_path = r"C:\Users\CL-11\OneDrive\Repos\agency-swarm-lab\CausalityAnalysisAgency\AA.csv"

    print("Available Agency methods:")
    print([method for method in dir(agency) if not method.startswith('_')])

    '''agency.initiate_chat([{
        "role": "user",
        "content": f"Clean the data from file {file_path} using mean strategy"
    }])'''

    response = agency.get_completion(f"Clean the data from file {file_path} using mean strategy")
    print("Response:", response)



if __name__ == '__main__':
    #agency.demo_gradio()
    main()