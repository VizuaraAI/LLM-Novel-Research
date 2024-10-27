from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client.files.create(
  file=open("data-001.jsonl", "rb"),
  purpose="fine-tune"
)