import autogen
from autogen import ConversableAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from dotenv import load_dotenv
import os
import chromadb  # Assuming ChromaDB is being used for retrieval

# Load environment variables from the .env file
load_dotenv('C:\\Users\\Raymundoneo\\Documents\\LLM-Novel-research project\\LLM-Novel-Research\\config\\.env')

# Get the API key
api_key = os.getenv('API_KEY')

# Configuration for LLM model
config_list = [
    {
        'model': 'gpt-3.5-turbo',
        'api_key': api_key
    }
]

llm_config = {
    "seed": 42,
    "config_list": config_list,
    "temperature": 0
}

# Assistant Agent (LLM) to generate code for SIR model numerically
assistant = ConversableAgent(
    "LLM",
    system_message="You will produce Julia code to solve equations numerically. The Julia code must be compatible with execution from a Python kernel using the `julia` Python package. "
                  "Always use the `Main` module from `julia` to execute Julia code within Python. Start by importing `Main` from the `julia` package in Python. "
                  "If a required Julia library is required, only import it. It is already installed. "
                  "All Julia code should be passed as strings to `Main.eval` and must handle numerical computations or library installations. "
                  "Whenever possible, include Python comments explaining each step. "
                  "All the code you write should be already executable from the python terminal. Do not instruct the user to modify or adapt the code you provide before executing it. "
                  "Importantly, the code you generate will be executed as a temporary file. Therefore, if you suggest new or supplementing code after feedback from the user, append it to the previous code because there is no track in the terminal of the previous executions. "
                  "Once you generate the code to solve the prompt handled by the user, you will add to it a code block so that when the user runs all the script, the Julia code for solving the prompt is stored in the user's directory as a `.jl` file. "
                  "When the user states that your solution is successful or when the output code is successful, then say TERMINATE",
    llm_config=llm_config
)

# RetrieveUserProxyAgent for Neural ODE retrieval
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/SciML/DiffEqFlux.jl/master/docs/src/examples/neural_ode.md",  # Link to Neural ODE documentation
        ],
        "model": config_list[0]["model"],
        "vector_db": "chroma",
        "overwrite": False,  # set to True if you want to overwrite an existing collection
        "get_or_create": True,

        #"custom_text_types": ["mdx"],  # Assuming Markdown format
        "chunk_token_size": 2000,  # ChromaDB client for retrieval
        #"embedding_model": "all-mpnet-base-v2",  # Embedding model
        #"get_or_create": True,  # Create new collection if not found
    },
    code_execution_config=False  # Code execution is disabled for this agent, as it only retrieves information
)

# AI Agent (User Proxy) that runs Julia code
user_proxy = ConversableAgent(
    "CODE_executor",
    system_message="You will run Julia Code within Python for solving equations numerically.",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "code", "use_docker": False},
    llm_config=llm_config
)

code_reviewer = ConversableAgent(
    "CODE_reviewer",
    system_message=(
        "You are a code reviewer specializing in reviewing and adapting Julia code to ensure it can run seamlessly within a Python kernel. "
        "The code you provide or modify must follow this format:\n\n"
        "```python\n"
        "from julia import Main\n\n"
        "# Execute the entire Julia script as a multi-line string\n"
        "result = Main.eval(\"\"\"\n"
        "    # The Julia-generated code\n"
        "\"\"\")\n\n"
        "print(result)  # Outputs the result of the Julia computation\n"
        "```\n\n"
        "Your tasks include:\n"
        "- Make sure all the needed libraries are call in the script: using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,OptimizationOptimisers, Random, Plots"
        "- Ensure Dense is called from Lux (Lux.Dense()). \n"
        "- Ensure that you explicitly call Chain from the Lux library. Change Chain to Lux.Chain(). \n"
        "- Ensure the random number generator is set to rng = MersenneTwister(99). \n "
        "- Ensure the final plot is saved in the current directory."
        "- Ensuring the Julia code is syntactically correct and executable within a Python kernel using the `Main` module.\n"
        "- Adapting the Julia code to run as a complete script within the `Main.eval` block.\n"
        "- Checking that all required Julia packages are explicitly imported in the code.\n"
        "- Ensuring there are no package installation commands (e.g., `Pkg.add` or similar) present in the code.\n"
        "- Providing constructive feedback to enhance code clarity, efficiency, and maintainability.\n"
        "- Highlighting and addressing any compatibility or execution issues when running the Julia code from a Python kernel."
    ),
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    llm_config=llm_config
)




# Task 1: Numerically solving the SIR Model using standard numerical methods (Euler or Runge-Kutta)
#task_1 = """Solve the SIR Model numerically using Julia DifferentialEquations and Plots libraries."""

#user_proxy.initiate_chat(
 #   assistant,
  #  message=task_1
#)
# Task 2: Solve the SIR Model using Neural ODE framework (using DiffEqFlux.jl)
code_problem = """Solve the SIR Model using a Neural ODE framework in Julia. Train the neural network using LUX to approximate the SIR model and plot the results."""


#chat_result = ragproxyagent.initiate_chat(
 #   assistant, message=ragproxyagent.message_generator, problem=code_problem
#)

# **Task 1**: Directly solving the SIR model numerically (handled by the LLM agent)


# **Task 2**: Use `ragproxyagent` to retrieve relevant Neural ODE info and generate code for Neural ODE
#ragproxyagent.initiate_chat(
 #   assistant,
  #  message=task_2,
   # search_string="neural_ode"
#)  Searching for relevant information related to Neural ODE and DiffEqFlux

# The `ragproxyagent` will retrieve information about Neural ODE, which will be used by the assistant to generate the necessary code for Task 2.

group_chat = autogen.GroupChat(
    agents=[user_proxy, ragproxyagent, assistant, code_reviewer], messages=[], max_round=12
)
manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

ragproxyagent.initiate_chat(
    manager,
    message=ragproxyagent.message_generator, problem=code_problem
)
