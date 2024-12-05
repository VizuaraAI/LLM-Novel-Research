import autogen
from autogen import ConversableAgent
#Importing files to avoid showing the API key
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv('C:\\Users\\Raymundoneo\\Documents\\LLM-Novel-research project\\LLM-Novel-Research\\config\\.env')


# Get the API key
api_key = os.getenv('API_KEY')



config_list = [
    {
        'model': 'gpt-4',
        'api_key': api_key
    }
]

llm_config = {
    "seed":42,
    "config_list": config_list,
    "temperature":0


}
#Assistant, representing the llm
assistant = ConversableAgent(
    "LLM",
    system_message="You will produce Julia Code for solving equations numerically.",
    llm_config = llm_config
)

#AI agent, representing the USER.
user_proxy = ConversableAgent(
     "CODE_executor",
    system_message="You will run Julia Code within python for solving equations numerically. In the case a Julia library is missing you should implement 'from julia import Pkg,Pkg.add(PackageName)'",
    #name = "user_proxy",
    human_input_mode = "ALWAYS",
    max_consecutive_auto_reply = 10,
    is_termination_msg = lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "code", "use_docker": False},
    llm_config = llm_config,
    #system_message = """Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet """

)
task = """Solve the SIR Model numerically using Julia language."""

user_proxy.initiate_chat(
    assistant, 
    message = task
)


