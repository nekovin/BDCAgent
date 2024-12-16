from agency_swarm.agents import Agent


class ResultInterpreter(Agent):
    def __init__(self):
        super().__init__(
            name="ResultInterpreter",
            description="The ResultInterpreter agent interprets results from the CausalityAnalyzer and presents them in a user-friendly format, providing insights and explanations about the causal relationships found in the data.",
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
