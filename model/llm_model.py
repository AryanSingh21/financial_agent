from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

model = GroqModel(
    'llama-3.3-70b-versatile', 
    provider=GroqProvider(api_key='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')
)

# Provided the key here for testing purpose and considering the fact that, the team will be running the code
# This key will be revoked after few days