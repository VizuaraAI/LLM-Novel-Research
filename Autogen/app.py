import autogen
from autogen import ConversableAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from dotenv import load_dotenv
import os
import chromadb  # Assuming ChromaDB is being used for retrieval
from autogen.agentchat import GroupChat, GroupChatManager
from autogen.oai.openai_utils import config_list_from_dotenv

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
                  "Make sure the model you are solving is the one asked by the user. Importantly, adapt the code for the specific model. The context gives you an example but needs to be adapted."
                  "All the code you write should be already executable from the python terminal. Do not instruct the user to modify or adapt the code you provide before executing it. "
                  "Importantly, the code you generate will be executed as a temporary file. Therefore, if you suggest new or supplementing code after feedback from the user, append it to the previous code because there is no track in the terminal of the previous executions. "
                  "Once you generate the code to solve the prompt handled by the user, you will add to it a code block so that when the user runs all the script, the Julia code for solving the prompt is stored in the user's directory as a `.jl` file. "
                  "If you get a message saying that you need to update the training parameters review the code script to change the learning rates, and epochs. Then send the FULL code script to the next agent.",               
    llm_config=llm_config,
    description="""I am **ONLY** allowed to speak **immediately** after `ragproxyagent` and `training_reviewer`. When `training_reviewer` talks to me, I will review the previous code script I produce to change the training parameters. The next agent to speak is `CODE_reviewer`."""
)

# RetrieveUserProxyAgent for Neural ODE retrieval
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/Raymundv/DiffEqFlux.jl/refs/heads/master/docs/src/examples/neural_ode.md",  # Link to Neural ODE documentation
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
    llm_config=llm_config,
    description="""I am **ONLY** allowed to speak **immediately** after `CODE_reviewer`. If the code can not run due to a syntax problem or logical problem, the next agent to speak will be `CODE_reviewer`. Only If I encounter a timeout problem or the last number on the log is greater than 10e-5, the next agent to speak will be `training_reviewer`"""
)

code_reviewer = ConversableAgent(
    "CODE_reviewer",
    system_message=(
        "You are a code reviewer specializing in reviewing and adapting Julia code to ensure it can run seamlessly within a Python kernel. "
        "You should always take the code provided by the assistant agent, and modify it to follow this format:\n\n"
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
        "- Highlighting and addressing any compatibility or execution issues when running the Julia code from a Python kernel. "
        "-"
       
    ),
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    llm_config=llm_config,
    description="""I am **ONLY** allowed to speak **immediately** after `LLM` and `CODE_executor`. I will take the Code given by LLM and send it to CODE_executor"""
)

training_reviewer = ConversableAgent(
    "training_reviewer",
    system_message=(
        "You are a Training Reviewer Agent responsible for monitoring the training process during the execution of optimization algorithms. "
            "Your tasks include the following:\n\n"
            "1. **Monitoring Training Loss**: Check if the final loss exceeds the specified threshold (default is 0.001). "
            "2. You will always critic the training routine performance when the final loss is below the 0.001 threshold. Always double check that the final loss is greater than 0.001."
            "If the final loss is greater than the threshold, suggest changes to training parameters to help reduce the loss.\n\n"
            "2. **Timeout Monitoring**: If a timeout message is encountered, suggest adjustments to the number of epochs or optimization parameters to avoid timeouts in the future.\n\n"
            "3. **Suggesting Training Parameters**: Based on the loss or timeout, you should suggest the following training adjustments:\n"
            "   - If the final loss is greater than 0.001, suggest increasing the initial learning rate or step size, adjusting the batch size, and increasing the number of epochs\n"
            "   - If a timeout occurs, suggest increasing the number of epochs and modifying the optimizer parameters (e.g., using ADAM or BFGS). Consider the current model's configuration and training environment.\n\n"
            "4. Finally, and more importantly, based on your findings, if the loss is greater than the threshold or there is a timeout error, you will prompt a message stating that the code should be reviewed with changed optimization routine parameters."
    ),
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    llm_config=llm_config
)

#Defining the workflow transitions
graph_dict = {}
graph_dict[ragproxyagent] = [assistant]
graph_dict[assistant] = [code_reviewer]
graph_dict[code_reviewer] = [user_proxy]
graph_dict[user_proxy] = [code_reviewer, training_reviewer]
graph_dict[training_reviewer] = [assistant]

# Task 2: Solve the SIR Model using Neural ODE framework (using DiffEqFlux.jl)
code_problem = """Solve the SIR Model over a timespan from 0 to 100 (50 points), using a Neural ODE framework in Julia."""


#Defining the agents
#
agents=[user_proxy, ragproxyagent, assistant, code_reviewer, training_reviewer] 



# create the groupchat
group_chat = GroupChat(agents=agents, messages=[], max_round=25, allowed_or_disallowed_speaker_transitions=graph_dict, allow_repeat_speaker=None, speaker_transitions_type="allowed")

# create the manager
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
)
ragproxyagent.initiate_chat(
    manager,
    message=ragproxyagent.message_generator, problem=code_problem
)
