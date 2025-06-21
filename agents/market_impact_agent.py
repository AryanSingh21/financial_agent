from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic import BaseModel
from typing import List, Dict, Any
import json
from datetime import datetime
from model.llm_model import model


class MarketImpactAssessment(BaseModel):
    impact_score: float
    affected_sectors: List[str]
    potential_stock_movement: str 
    time_horizon: str
    key_factors: List[str]
    risk_level: str

market_impact_agent = Agent(
    model,
    output_type=MarketImpactAssessment,
    system_prompt=(
        'You are a market impact assessment specialist. Evaluate how financial news might affect '
        'markets, sectors, and individual stocks. Consider historical patterns, market conditions, '
        'and potential ripple effects. Assess both immediate and longer-term implications. '
        'Provide specific sectors that might be affected and explain your reasoning.'
    ),
)