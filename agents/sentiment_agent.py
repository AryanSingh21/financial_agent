from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic import BaseModel
from typing import List, Dict, Any
import json
from datetime import datetime
from model.llm_model import model

class SentimentAnalysis(BaseModel):
    sentiment_score: float
    confidence: float
    key_phrases: List[str]
    overall_sentiment: str
    reasoning: str

sentiment_analysis_agent = Agent(
    model,
    output_type=SentimentAnalysis,
    system_prompt=(
        'You are a specialist in financial sentiment analysis. Analyze the given financial news content '
        'and provide detailed sentiment metrics. Focus on market-relevant language, corporate terminology, '
        'and investor sentiment indicators. Consider both explicit statements and implicit implications. '
        'Provide specific key phrases that influenced your analysis.'
    ),
)