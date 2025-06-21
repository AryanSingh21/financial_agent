from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
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

class MarketImpactAssessment(BaseModel):
    impact_score: float
    affected_sectors: List[str]
    potential_stock_movement: str 
    time_horizon: str
    key_factors: List[str]
    risk_level: str 


class FinancialNewsAnalysis(BaseModel):
    article_id: str
    sentiment_analysis: SentimentAnalysis
    market_impact: MarketImpactAssessment
    overall_recommendation: str
    confidence_level: float

# Initialize the model




financial_analysis_agent = Agent(
    model,
    output_type=FinancialNewsAnalysis,
    system_prompt=(
        'You are a senior financial analyst orchestrating a comprehensive analysis of financial news. '
        'Use the sentiment_analyzer to get detailed sentiment analysis, then use the market_impact_assessor '
        'to evaluate potential market effects. Combine both analyses to provide a final recommendation. '
        'Be thorough, objective, and consider multiple perspectives.'
    ),
)