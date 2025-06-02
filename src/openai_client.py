import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

# Instantiate the v1 OpenAI client pointed at your Azure/Secure proxy
client = OpenAI(
    api_key      = os.getenv("AZURE_OPENAI_API_KEY"),
    api_type     = "azure",
    api_base     = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version  = os.getenv("OPENAI_API_VERSION"),
)

# Export your deployment names
CHAT_ENGINE  = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_ENGINE = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "your-embedding-deployment-name")
