import os
from dotenv import load_dotenv

load_dotenv()

SUBSCRIPTION_KEY = os.getenv("GENAIPLATFORM_FARM_SUBSCRIPTION_KEY")

if not SUBSCRIPTION_KEY:
    raise ValueError(
        "GENAIPLATFORM_FARM_SUBSCRIPTION_KEY is not set. Please add it to .env"
    )

BOSCH_URL = (
    "https://aoai-farm.bosch-temp.com/api/openai/deployments/"
    "askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18/chat/completions"
    "?api-version=2024-08-01-preview"
)

PROXIES = {
    "http": "http://127.0.0.1:3128",
    "https": "http://127.0.0.1:3128",
}

COURSE_NAME = "Introduction to Data_and_Data_Science"
CHAPTERS = [
    "Analysis vs Analytics",
    "Programming Languages & Software Employed in Data Science - All the Tools You Need",
]

# Transcript is ~8260 characters, so these are sufficient and efficient
RETRIEVER_TOP_K = 4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Keep answer/summarization bounded
MODEL_MAX_TOKENS = 1200
MODEL_TEMPERATURE = 0.2

# Summarize once conversation gets a bit longer
SUMMARY_TRIGGER_MESSAGE_COUNT = 6
KEEP_LAST_MESSAGES = 2
ANSWER_RECENT_MESSAGES = 4