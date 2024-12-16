from agency_swarm.agents import Agent


class CausalityCEO(Agent):
    def __init__(self):
        super().__init__(
            name="CausalityCEO",
            description="The CausalityCEO agent oversees the entire process of the Causality Analysis Agency. It coordinates between agents, ensures the agency's mission is achieved, and communicates with the user and other agents to manage tasks and report results.",
            instructions="./instructions.md",
            files_folder="./files",
            schemas_folder="./schemas",
            tools=[],
            tools_folder="./tools",
            temperature=0.3,
            max_prompt_tokens=25000,
        )

    def response_validator(self, message):
        return message
