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
    system_message="You will produce Julia code to solve equations numerically. The Julia code must be compatible with execution from a Python kernel using the `julia` Python package. "
        "Always use the `Main` module from `julia` to execute Julia code within Python. Start by importing `Main` from the `julia` package in Python. "
        "If a required Julia library is required, only import it. It is already installed."
        "All Julia code should be passed as strings to `Main.eval` and must handle numerical computations or library installations. "
        "Whenever possible, include Python comments explaining each step. "
        "All the code you write, should be already executable from the python terminal. Do not instruct the user to modify or adapt the code you provide before executing it."
        "Importantly, the code you generate will be executed as a temporary file. Therefore, if you suggest new or supplementing code after feedback from the user, append it to the previous code because there is no track in the terminal of the previous executions."
        "Once you generate the code to solve the prompt handled by the user, you will add to it a code block so that when the user runs all the script, the Julia code for solving the prompt is stored in the user's directory as a `.jl` file."
        "When the user states that your solution is succesful or when the output code is succesful then say TERMINATE",
    llm_config = llm_config
)

#AI agent, representing the USER.
user_proxy = ConversableAgent(
     "CODE_executor",
    system_message="You will run Julia Code within python for solving equations numerically.",
    #name = "user_proxy",
    human_input_mode = "ALWAYS",
    max_consecutive_auto_reply = 10,
    is_termination_msg = lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "code", "use_docker": False},
    llm_config = llm_config,
    #system_message = """Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet """

)
task = """Solve the SIR Model numerically using Julia language and save the plot."""

user_proxy.initiate_chat(
    assistant, 
    message = task
)




#Implementing the Neural ODE routing

import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from autogen import ConversableAgent


# Step 2: Implementing RAG - Retrieving Information from DiffEqFlux Docs
def fetch_diff_eq_flux_content():
    url = "https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the relevant sections of the page - example: getting all paragraphs
    paragraphs = soup.find_all('p')
    content = "\n".join([para.get_text() for para in paragraphs])

    # You could refine this to capture more relevant content like code blocks, etc.
    return content

# Step 3: Task 2 - Use the content retrieved to continue and augment the task
def task_2_with_rag():
    # Retrieve content from DiffEqFlux Neural ODE example
    rag_content = fetch_diff_eq_flux_content()

    # New task 2 message including the RAG content
    task_2_message = f"""
    Now that you have solved the SIR model numerically, I want you to proceed with implementing a neural ODE model using the Neural ODE example from DiffEqFlux.
    Here is some additional information to help guide you:

    {rag_content}

    Start by implementing the Neural ODE as described in the DiffEqFlux docs.
    """

    # Continue the chat with task_2, building upon previous memory (task_1)
    user_proxy.initiate_chat(
        assistant,
        message=task_2_message
    )

# Execute Task 2 after Task 1 is completed
task_2_with_rag()

