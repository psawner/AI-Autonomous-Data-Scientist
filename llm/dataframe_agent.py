from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()

model = init_chat_model(
    "groq:meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)


def create_dataframe_agent(df):

    agent = create_pandas_dataframe_agent(
        model,
        df,
        verbose=False,
        allow_dangerous_code=True
    )

    return agent