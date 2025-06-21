from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic import BaseModel
from typing import List, Dict, Any
import json
from datetime import datetime
from agents.financial_agent import financial_analysis_agent
from agents.sentiment_agent import sentiment_analysis_agent
from agents.market_impact_agent import market_impact_agent
# Define output models
class SentimentAnalysis(BaseModel):
    sentiment_score: float  # -1.0 to 1.0 (negative to positive)
    confidence: float  # 0.0 to 1.0
    key_phrases: List[str]
    overall_sentiment: str  # "positive", "negative", "neutral"
    reasoning: str

class MarketImpactAssessment(BaseModel):
    impact_score: float  # 0.0 to 10.0 (low to high impact)
    affected_sectors: List[str]
    potential_stock_movement: str  # "up", "down", "neutral"
    time_horizon: str  # "immediate", "short-term", "long-term"
    key_factors: List[str]
    risk_level: str  # "low", "medium", "high"

class FinancialNewsAnalysis(BaseModel):
    article_id: str
    sentiment_analysis: SentimentAnalysis
    market_impact: MarketImpactAssessment
    overall_recommendation: str
    confidence_level: float

# Tool for sentiment analysis
@financial_analysis_agent.tool
async def sentiment_analyzer(ctx: RunContext[None], headline: str, content: str, article_id: str) -> SentimentAnalysis:
    """Analyze the sentiment of financial news content"""
    prompt = f"""
    Analyze the sentiment of this financial news:
    
    Article ID: {article_id}
    Headline: {headline}
    Content: {content}
    
    Provide detailed sentiment analysis with scores, confidence levels, and key phrases.
    """
    
    result = await sentiment_analysis_agent.run(
        prompt,
        usage=ctx.usage,
    )
    print(f"Sentiment Analysis Result: {result.output}")
    return result.output

# Tool for market impact assessment
@financial_analysis_agent.tool
async def market_impact_assessor(ctx: RunContext[None], headline: str, content: str, 
                                sentiment_data: SentimentAnalysis, article_id: str) -> MarketImpactAssessment:
    """Assess the potential market impact of financial news"""
    prompt = f"""
    Assess the market impact of this financial news:
    
    Article ID: {article_id}
    Headline: {headline}
    Content: {content}
    
    Sentiment Context:
    - Overall Sentiment: {sentiment_data.overall_sentiment}
    - Sentiment Score: {sentiment_data.sentiment_score}
    - Key Phrases: {', '.join(sentiment_data.key_phrases)}
    - Reasoning: {sentiment_data.reasoning}
    
    Evaluate potential market impact, affected sectors, and provide specific recommendations.
    """
    
    result = await market_impact_agent.run(
        prompt,
        usage=ctx.usage,
    )
    print(f"Market Impact Assessment Result: {result.output}")
    return result.output

# Sample financial news data
sample_news = {
    "article_id": "FIN-003",
    "headline": "Amazon announces 'transformational' AI venture, but at massive cost",
    "content": "Amazon (NASDAQ: AMZN) unveiled Project Olympus, a $50 billion investment in AGI de",
    "published_at": "2024-09-15T09:00:00Z"
}


def run_financial_analysis(news_data: Dict[str, Any]) -> FinancialNewsAnalysis:
    """Run the complete financial news analysis"""
    prompt = f"""
    Analyze this financial news comprehensively:
    
    Article ID: {news_data['article_id']}
    Headline: {news_data['headline']}
    Content: {news_data['content']}
    Published: {news_data['published_at']}
    
    First get sentiment analysis, then market impact assessment, and finally provide 
    your overall recommendation with confidence level.
    """
    
    result = financial_analysis_agent.run_sync(
        prompt,
        usage_limits=UsageLimits(request_limit=15),
    )
    
    return result

# Run the analysis
if __name__ == "__main__":
    print("Starting Financial News Analysis...")
    print("=" * 50)
    
    # Run analysis on sample news
    analysis_result = run_financial_analysis(sample_news)
    
    print("\nFINAL ANALYSIS RESULT:")
    print("=" * 50)
    print(f"Article ID: {analysis_result.output.article_id}")
    print(f"Overall Recommendation: {analysis_result.output.overall_recommendation}")
    print(f"Confidence Level: {analysis_result.output.confidence_level:.2f}")
    
    print("\nSENTIMENT ANALYSIS:")
    print("-" * 30)
    sentiment = analysis_result.output.sentiment_analysis
    print(f"Sentiment: {sentiment.overall_sentiment}")
    print(f"Score: {sentiment.sentiment_score:.2f}")
    print(f"Confidence: {sentiment.confidence:.2f}")
    print(f"Key Phrases: {', '.join(sentiment.key_phrases)}")
    print(f"Reasoning: {sentiment.reasoning}")
    
    print("\nMARKET IMPACT ASSESSMENT:")
    print("-" * 30)
    impact = analysis_result.output.market_impact
    print(f"Impact Score: {impact.impact_score:.1f}/10")
    print(f"Affected Sectors: {', '.join(impact.affected_sectors)}")
    print(f"Potential Stock Movement: {impact.potential_stock_movement}")
    print(f"Time Horizon: {impact.time_horizon}")
    print(f"Risk Level: {impact.risk_level}")
    print(f"Key Factors: {', '.join(impact.key_factors)}")
    
    print(f"\nTOTAL USAGE:")
    print("-" * 30)
    print(analysis_result.usage())
    
    # Save analysis to file for conversation history
    with open('ai_chat_history.txt', 'a') as f:
        f.write("Financial News Analysis - AI Chat History\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.write("INPUT NEWS ARTICLE:\n")
        f.write("-" * 20 + "\n")
        f.write(json.dumps(sample_news, indent=2) + "\n\n")
        f.write("ANALYSIS RESULT:\n")
        f.write("-" * 20 + "\n")
        f.write(json.dumps(analysis_result.output.model_dump(), indent=2) + "\n\n")
        f.write("USAGE STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(str(analysis_result.usage()) + "\n\n\n")
    
    print("\nAnalysis saved to ai_chat_history.txt")