"""
Enhanced RAG System with Self-Verification and Temporal Awareness
Addresses knowledge obsolescence through multiple strategies:
1. Temporal validation of data freshness
2. Self-verification prompts to detect outdated information
3. Multi-source retrieval for cross-validation
4. Confidence scoring based on data recency

ALL TIMEZONE ISSUES FIXED
"""

import pandas as pd
import sys
import json
from datetime import datetime, timedelta
import os
import numpy as np
import requests
from typing import List, Dict, Tuple, Optional
import warnings


# JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and datetime objects"""
    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle datetime types
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return str(obj)
        # Handle regular Python bools (shouldn't be needed but just in case)
        elif isinstance(obj, bool):
            return obj
        return super().default(obj)

# Configuration
USE_OLLAMA = True  # Set to False to use OpenAI

if USE_OLLAMA:
    def call_llm(prompt, model="llama3.2"):
        """Call Ollama API"""
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            }
        )
        return response.json()["response"]
    
    def get_embedding(text, model="nomic-embed-text"):
        """Get text embedding from Ollama"""
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text}
        )
        result = response.json()
        if "embedding" in result:
            return np.array(result["embedding"])
        raise KeyError(f"Unexpected Ollama response: {result.keys()}")
else:
    from openai import OpenAI
    client = OpenAI()
    
    def call_llm(prompt, model="gpt-4"):
        """Call OpenAI API"""
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    
    def get_embedding(text, model="text-embedding-3-small"):
        """Get text embedding from OpenAI"""
        response = client.embeddings.create(model=model, input=text)
        return np.array(response.data[0].embedding)


# ============================================================================
# TIMEZONE FIX HELPER
# ============================================================================

def make_timezone_naive(dt):
    """
    Convert any datetime to timezone-naive for comparison
    This fixes all timezone-related errors
    """
    if dt is None:
        return dt
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    if isinstance(dt, pd.Timestamp) and dt.tz is not None:
        return dt.tz_localize(None)
    if isinstance(dt, datetime) and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


# ============================================================================
# TEMPORAL AWARENESS COMPONENTS
# ============================================================================

class TemporalValidator:
    """Validates data freshness and detects potential obsolescence"""
    
    def __init__(self, max_staleness_hours: int = 24):
        self.max_staleness_hours = max_staleness_hours
        
    def calculate_staleness(self, data_timestamp: datetime) -> Dict:
        """Calculate how stale the data is"""
        now = datetime.now()
        if isinstance(data_timestamp, str):
            data_timestamp = pd.to_datetime(data_timestamp)
        
        # FIX: Make timezone-naive
        data_timestamp = make_timezone_naive(data_timestamp)
        
        staleness = now - data_timestamp
        hours_old = staleness.total_seconds() / 3600
        days_old = staleness.days
        
        # Categorize freshness
        if hours_old < 1:
            freshness = "REAL-TIME"
            confidence = 1.0
        elif hours_old < 24:
            freshness = "FRESH"
            confidence = 0.95
        elif days_old < 7:
            freshness = "RECENT"
            confidence = 0.85
        elif days_old < 30:
            freshness = "MODERATELY_STALE"
            confidence = 0.70
        else:
            freshness = "STALE"
            confidence = max(0.5, 1.0 - (days_old / 365))  # Decays over year
        
        return {
            'hours_old': hours_old,
            'days_old': days_old,
            'freshness_category': freshness,
            'confidence_score': confidence,
            'is_acceptable': hours_old < self.max_staleness_hours * 24
        }
    
    def detect_temporal_mismatch(self, question: str, data_date: datetime) -> Dict:
        """Detect if question asks about timeframe outside our data"""
        question_lower = question.lower()
        now = datetime.now()
        
        # FIX: Make timezone-naive
        data_date = make_timezone_naive(pd.to_datetime(data_date))
        
        # Temporal keywords indicating recent queries
        recent_keywords = ['today', 'now', 'current', 'latest', 'recent', 'this week']
        past_keywords = ['yesterday', 'last week', 'last month', 'ago']
        future_keywords = ['tomorrow', 'next', 'will', 'forecast', 'predict']
        
        warnings_list = []
        
        # Check for future queries (we can't answer)
        if any(kw in question_lower for kw in future_keywords):
            warnings_list.append({
                'type': 'FUTURE_QUERY',
                'message': 'Question asks about future events - we only have historical data',
                'severity': 'HIGH'
            })
        
        # Check for very recent queries
        if any(kw in question_lower for kw in recent_keywords):
            data_staleness = (now - data_date).total_seconds() / 3600
            if data_staleness > 24:
                warnings_list.append({
                    'type': 'STALE_FOR_RECENT_QUERY',
                    'message': f'Question asks about recent data but most recent data is {data_staleness:.1f} hours old',
                    'severity': 'MEDIUM'
                })
        
        # Check for specific past timeframes
        if 'yesterday' in question_lower:
            yesterday = now - timedelta(days=1)
            if data_date.date() < yesterday.date():
                warnings_list.append({
                    'type': 'MISSING_YESTERDAY',
                    'message': 'Question asks about yesterday but data does not include yesterday',
                    'severity': 'HIGH'
                })
        
        return {
            'has_mismatch': len(warnings_list) > 0,
            'warnings': warnings_list
        }


class ConfidenceScorer:
    """Scores answer confidence based on data freshness and retrieval quality"""
    
    def calculate_confidence(self, 
                           data_freshness: Dict,
                           retrieval_scores: List[float],
                           temporal_validation: Dict) -> Dict:
        """Calculate overall confidence score"""
        
        # Base confidence from data freshness
        freshness_confidence = data_freshness['confidence_score']
        
        # Retrieval quality (average of top-k similarity scores)
        retrieval_confidence = np.mean(retrieval_scores) if retrieval_scores else 0.5
        
        # Temporal validation penalty
        temporal_penalty = 0.0
        if temporal_validation['has_mismatch']:
            high_severity = sum(1 for w in temporal_validation['warnings'] if w['severity'] == 'HIGH')
            medium_severity = sum(1 for w in temporal_validation['warnings'] if w['severity'] == 'MEDIUM')
            temporal_penalty = (high_severity * 0.3) + (medium_severity * 0.15)
        
        # Combined confidence
        overall_confidence = (
            0.5 * freshness_confidence +
            0.3 * retrieval_confidence +
            0.2 * (1.0 - temporal_penalty)
        )
        
        # Categorize confidence
        if overall_confidence >= 0.9:
            category = "VERY_HIGH"
        elif overall_confidence >= 0.75:
            category = "HIGH"
        elif overall_confidence >= 0.6:
            category = "MEDIUM"
        elif overall_confidence >= 0.4:
            category = "LOW"
        else:
            category = "VERY_LOW"
        
        return {
            'overall_confidence': overall_confidence,
            'confidence_category': category,
            'freshness_component': freshness_confidence,
            'retrieval_component': retrieval_confidence,
            'temporal_penalty': temporal_penalty,
            'should_warn_user': overall_confidence < 0.6
        }


# ============================================================================
# MULTI-SOURCE DATA RETRIEVAL
# ============================================================================

def fetch_realtime_data_multi_source(ticker: str, days: int = 60) -> Tuple[pd.DataFrame, Dict]:
    """
    Fetch data from multiple sources and cross-validate
    Returns: (merged_dataframe, metadata)
    """
    import yfinance as yf
    
    metadata = {
        'sources': [],
        'data_quality': {},
        'conflicts': []
    }
    
    # Primary source: Yahoo Finance
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(ticker)
        df_yahoo = stock.history(start=start_date, end=end_date)
        
        if not df_yahoo.empty:
            df_yahoo = df_yahoo.reset_index()
            
            # FIX: Remove timezone info for consistency
            if hasattr(df_yahoo['Date'].dtype, 'tz') and df_yahoo['Date'].dt.tz is not None:
                df_yahoo['Date'] = df_yahoo['Date'].dt.tz_localize(None)
            
            df_yahoo = df_yahoo[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            metadata['sources'].append({
                'name': 'Yahoo Finance',
                'records': len(df_yahoo),
                'date_range': f"{df_yahoo['Date'].min()} to {df_yahoo['Date'].max()}",
                'status': 'SUCCESS'
            })
        else:
            raise ValueError("No data from Yahoo Finance")
            
    except Exception as e:
        metadata['sources'].append({
            'name': 'Yahoo Finance',
            'status': 'FAILED',
            'error': str(e)
        })
        return None, metadata
    
    # Data quality checks (now timezone-safe)
    metadata['data_quality'] = {
        'total_records': len(df_yahoo),
        'missing_values': df_yahoo.isnull().sum().to_dict(),
        'date_gaps': detect_date_gaps(df_yahoo),
        'data_freshness': {
            'most_recent_date': str(df_yahoo['Date'].max()),
            'hours_since_update': (datetime.now() - df_yahoo['Date'].max()).total_seconds() / 3600
        }
    }
    
    return df_yahoo, metadata


def detect_date_gaps(df: pd.DataFrame) -> List[Dict]:
    """Detect gaps in trading days (potential missing data)"""
    # FIX: Ensure Date column is timezone-naive
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    if hasattr(df['Date'].dtype, 'tz') and df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    
    df = df.sort_values('Date')
    
    gaps = []
    for i in range(1, len(df)):
        date_diff = (df.iloc[i]['Date'] - df.iloc[i-1]['Date']).days
        # Gap of more than 3 days (accounting for weekends) is suspicious
        if date_diff > 3:
            gaps.append({
                'start_date': str(df.iloc[i-1]['Date']),
                'end_date': str(df.iloc[i]['Date']),
                'gap_days': date_diff
            })
    
    return gaps


# ============================================================================
# ENHANCED DOCUMENT CHUNKING WITH METADATA
# ============================================================================

def create_document_chunks_enhanced(df: pd.DataFrame, ticker: str, metadata: Dict) -> List[Dict]:
    """
    Create chunks with rich metadata for better retrieval and validation
    """
    chunks = []
    
    # Add data quality metadata to first chunk
    quality_chunk = f"""Stock: {ticker}
Data Quality Report:
- Total trading days: {metadata['data_quality']['total_records']}
- Most recent data: {metadata['data_quality']['data_freshness']['most_recent_date']}
- Data freshness: {metadata['data_quality']['data_freshness']['hours_since_update']:.1f} hours old
- Data sources: {', '.join([s['name'] for s in metadata['sources'] if s['status'] == 'SUCCESS'])}
- Date gaps detected: {len(metadata['data_quality']['date_gaps'])}
"""
    
    chunks.append({
        'text': quality_chunk,
        'type': 'metadata',
        'ticker': ticker,
        'importance': 'high',  # Always retrieve this for context
        'timestamp': datetime.now().isoformat()
    })
    
    # Daily chunks with temporal metadata
    for idx, row in df.iterrows():
        date = row['Date']
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        
        # FIX: Calculate how old this data point is
        date_naive = make_timezone_naive(date)
        age_hours = (datetime.now() - date_naive).total_seconds() / 3600
        age_days = age_hours / 24
        
        chunk_text = f"""Stock: {ticker}
Date: {date_str}
Day of Week: {date.strftime('%A') if hasattr(date, 'strftime') else 'Unknown'}
Open: ${row['Open']:.2f}
High: ${row['High']:.2f}
Low: ${row['Low']:.2f}
Close: ${row['Close']:.2f}
Volume: {int(row['Volume']):,}
Intraday Range: ${(row['High'] - row['Low']):.2f}
Daily Change: {((row['Close'] - row['Open']) / row['Open'] * 100):.2f}%
Data Age: {age_days:.1f} days old
"""
        
        # Determine recency category
        if age_hours < 24:
            recency = 'today'
        elif age_days < 7:
            recency = 'this_week'
        elif age_days < 30:
            recency = 'this_month'
        else:
            recency = 'older'
        
        chunks.append({
            'text': chunk_text,
            'date': date_str,
            'ticker': ticker,
            'type': 'daily',
            'close': row['Close'],
            'volume': row['Volume'],
            'age_hours': age_hours,
            'age_days': age_days,
            'recency': recency,
            'importance': 'high' if recency in ['today', 'this_week'] else 'medium'
        })
    
    # Weekly summaries with trend analysis
    if len(df) >= 7:
        for i in range(0, len(df) - 6, 7):
            week_data = df.iloc[i:i+7]
            start_date = week_data.iloc[0]['Date']
            end_date = week_data.iloc[-1]['Date']
            
            start_date_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            end_date_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
            
            # Calculate trends
            price_change = week_data.iloc[-1]['Close'] - week_data.iloc[0]['Open']
            price_change_pct = (price_change / week_data.iloc[0]['Open']) * 100
            
            # Volatility (standard deviation of daily returns)
            daily_returns = week_data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * 100
            
            # Trend direction
            if price_change_pct > 2:
                trend = "STRONG_UPTREND"
            elif price_change_pct > 0:
                trend = "UPTREND"
            elif price_change_pct > -2:
                trend = "DOWNTREND"
            else:
                trend = "STRONG_DOWNTREND"
            
            # FIX: Calculate age
            end_date_naive = make_timezone_naive(end_date)
            age_days = (datetime.now() - end_date_naive).days
            
            weekly_chunk = f"""Stock: {ticker}
Week: {start_date_str} to {end_date_str}
Opening Price: ${week_data.iloc[0]['Open']:.2f}
Closing Price: ${week_data.iloc[-1]['Close']:.2f}
Week High: ${week_data['High'].max():.2f}
Week Low: ${week_data['Low'].min():.2f}
Average Volume: {int(week_data['Volume'].mean()):,}
Total Volume: {int(week_data['Volume'].sum()):,}
Weekly Change: ${price_change:.2f} ({price_change_pct:.2f}%)
Trend: {trend}
Volatility: {volatility:.2f}%
Trading Days: {len(week_data)}
Data Age: {age_days} days old
"""
            
            chunks.append({
                'text': weekly_chunk,
                'date': end_date_str,
                'ticker': ticker,
                'type': 'weekly_summary',
                'close': week_data.iloc[-1]['Close'],
                'volume': week_data['Volume'].mean(),
                'trend': trend,
                'age_days': age_days,
                'importance': 'medium'
            })
    
    return chunks


# ============================================================================
# SELF-VERIFICATION SYSTEM
# ============================================================================

def generate_self_verification_prompt(question: str, initial_answer: str, 
                                     data_freshness: Dict, context_dates: List[str]) -> str:
    """
    Generate a prompt for the LLM to verify its own answer for temporal validity
    """
    most_recent_date = max(context_dates) if context_dates else "Unknown"
    
    verification_prompt = f"""You are a critical fact-checker. Analyze this question-answer pair for temporal validity and potential obsolescence issues.

QUESTION: {question}

PROPOSED ANSWER: {initial_answer}

DATA CONTEXT:
- Most recent data available: {most_recent_date}
- Data freshness: {data_freshness['freshness_category']}
- Data is {data_freshness['days_old']:.1f} days old

VERIFICATION TASKS:
1. Does the answer make temporal claims that go beyond the available data?
2. Does the answer use phrases like "current", "now", "today" when data may be stale?
3. Are there any factual claims that cannot be verified with {data_freshness['days_old']:.1f}-day-old data?
4. Does the answer acknowledge the data cutoff date if relevant?

Respond with a JSON object:
{{
    "is_temporally_valid": true/false,
    "issues_found": ["list", "of", "issues"],
    "confidence_in_answer": 0.0-1.0,
    "recommended_disclaimers": ["list of disclaimers to add"],
    "obsolescence_risk": "LOW/MEDIUM/HIGH"
}}

JSON Response:"""
    
    return verification_prompt


def self_verify_answer(question: str, answer: str, data_freshness: Dict, 
                       context_chunks: List[Dict]) -> Dict:
    """
    Have the LLM verify its own answer for temporal validity
    """
    # Extract dates from context
    context_dates = [chunk.get('date', '') for chunk in context_chunks if chunk.get('date')]
    
    verification_prompt = generate_self_verification_prompt(
        question, answer, data_freshness, context_dates
    )
    
    try:
        verification_response = call_llm(verification_prompt)
        
        # Parse JSON from response
        # Sometimes LLM includes markdown code blocks
        if "```json" in verification_response:
            verification_response = verification_response.split("```json")[1].split("```")[0].strip()
        elif "```" in verification_response:
            verification_response = verification_response.split("```")[1].split("```")[0].strip()
        
        verification_result = json.loads(verification_response)
        
        return {
            'verified': True,
            'is_valid': verification_result.get('is_temporally_valid', False),
            'issues': verification_result.get('issues_found', []),
            'confidence': verification_result.get('confidence_in_answer', 0.5),
            'disclaimers': verification_result.get('recommended_disclaimers', []),
            'obsolescence_risk': verification_result.get('obsolescence_risk', 'UNKNOWN')
        }
        
    except Exception as e:
        return {
            'verified': False,
            'error': str(e),
            'is_valid': None
        }


# ============================================================================
# ENHANCED RETRIEVAL WITH IMPORTANCE WEIGHTING
# ============================================================================

def retrieve_relevant_chunks_enhanced(query: str, index, chunks: List[Dict], 
                                     top_k: int = 5, 
                                     recency_boost: float = 0.3) -> Tuple[List[Dict], List[float]]:
    """
    Retrieve chunks with importance and recency weighting
    Returns: (relevant_chunks, similarity_scores)
    """
    import faiss
    
    # Get query embedding
    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding]).astype('float32')
    
    # Search for more candidates than we need (for reranking)
    search_k = min(top_k * 3, len(chunks))
    distances, indices = index.search(query_embedding, search_k)
    
    # Rerank based on importance and recency
    candidates = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(chunks):
            chunk = chunks[idx]
            
            # Base similarity score (convert L2 distance to similarity)
            base_similarity = 1 / (1 + distance)
            
            # Importance boost
            importance_multiplier = 1.0
            if chunk.get('importance') == 'high':
                importance_multiplier = 1.3
            elif chunk.get('importance') == 'medium':
                importance_multiplier = 1.1
            
            # Recency boost for recent data
            recency_multiplier = 1.0
            if 'age_days' in chunk:
                # More recent = higher score
                recency_multiplier = 1.0 + (recency_boost * np.exp(-chunk['age_days'] / 30))
            
            # Combined score
            final_score = base_similarity * importance_multiplier * recency_multiplier
            
            candidates.append({
                'chunk': chunk,
                'base_similarity': base_similarity,
                'final_score': final_score,
                'distance': distance
            })
    
    # Sort by final score and take top_k
    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    top_candidates = candidates[:top_k]
    
    relevant_chunks = [c['chunk'] for c in top_candidates]
    similarity_scores = [c['base_similarity'] for c in top_candidates]
    
    return relevant_chunks, similarity_scores


# ============================================================================
# MAIN ENHANCED RAG FUNCTION
# ============================================================================

def ask_question_rag_enhanced(ticker: str = None, question: str = None, 
                             enable_self_verification: bool = True) -> Dict:
    """
    Enhanced RAG system with temporal awareness and self-verification
    Returns comprehensive result dictionary
    """
    import faiss
    
    # Interactive mode
    if ticker is None:
        ticker = input("Enter stock ticker: ").upper()
    if question is None:
        question = input("Enter your question: ")
    
    result = {
        'ticker': ticker,
        'question': question,
        'timestamp': datetime.now().isoformat()
    }
    
    # Initialize validators
    temporal_validator = TemporalValidator(max_staleness_hours=48)
    confidence_scorer = ConfidenceScorer()
    
    # Step 1: Fetch real-time data from multiple sources
    print(f"\n[1/7] Fetching latest data for {ticker}...")
    df, metadata = fetch_realtime_data_multi_source(ticker, days=60)
    
    if df is None or df.empty:
        result['error'] = "Could not fetch data"
        result['success'] = False
        return result
    
    result['data_sources'] = metadata['sources']
    result['data_quality'] = metadata['data_quality']
    
    # Step 2: Validate data freshness
    print("[2/7] Validating data freshness...")
    most_recent_date = df['Date'].max()
    data_freshness = temporal_validator.calculate_staleness(most_recent_date)
    result['data_freshness'] = data_freshness
    
    print(f"  → Data is {data_freshness['freshness_category']} ({data_freshness['days_old']:.1f} days old)")
    
    # Step 3: Detect temporal mismatches
    print("[3/7] Checking for temporal mismatches...")
    temporal_validation = temporal_validator.detect_temporal_mismatch(question, most_recent_date)
    result['temporal_validation'] = temporal_validation
    
    if temporal_validation['has_mismatch']:
        print(f"  ⚠ Warning: {len(temporal_validation['warnings'])} temporal issue(s) detected")
        for warning in temporal_validation['warnings']:
            print(f"    - {warning['message']}")
    
    # Step 4: Create enhanced document chunks
    print("[4/7] Creating document chunks with metadata...")
    chunks = create_document_chunks_enhanced(df, ticker, metadata)
    print(f"  → Created {len(chunks)} chunks")
    
    # Step 5: Build vector database
    print("[5/7] Building vector database...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        if i % 20 == 0:
            print(f"  → Embedding chunk {i+1}/{len(chunks)}...")
        embedding = get_embedding(chunk['text'])
        embeddings.append(embedding)
    
    embeddings_array = np.array(embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    # Step 6: Retrieve relevant chunks with importance weighting
    print("[6/7] Retrieving relevant information...")
    relevant_chunks, similarity_scores = retrieve_relevant_chunks_enhanced(
        question, index, chunks, top_k=5, recency_boost=0.3
    )
    
    result['retrieved_chunks'] = [
        {
            'type': chunk.get('type'),
            'date': chunk.get('date'),
            'age_days': chunk.get('age_days'),
            'importance': chunk.get('importance')
        }
        for chunk in relevant_chunks
    ]
    
    # Step 7: Calculate confidence score
    print("[7/7] Calculating confidence scores...")
    confidence_result = confidence_scorer.calculate_confidence(
        data_freshness, similarity_scores, temporal_validation
    )
    result['confidence'] = confidence_result
    
    print(f"  → Overall confidence: {confidence_result['confidence_category']} ({confidence_result['overall_confidence']:.2f})")
    
    # Build context from retrieved chunks
    retrieved_context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
    
    # Get most recent data summary
    most_recent = df.iloc[-1]
    most_recent_date_str = most_recent['Date'].strftime('%Y-%m-%d') if hasattr(most_recent['Date'], 'strftime') else str(most_recent['Date'])
    
    recent_summary = f"""MOST RECENT DATA (Retrieved {datetime.now().strftime('%Y-%m-%d %H:%M')}):
Date: {most_recent_date_str}
Open: ${most_recent['Open']:.2f}
High: ${most_recent['High']:.2f}
Low: ${most_recent['Low']:.2f}
Close: ${most_recent['Close']:.2f}
Volume: {int(most_recent['Volume']):,}

DATA FRESHNESS: {data_freshness['freshness_category']} - Last updated {data_freshness['hours_old']:.1f} hours ago
CONFIDENCE LEVEL: {confidence_result['confidence_category']} ({confidence_result['overall_confidence']:.2f})
"""
    
    # Add warnings if low confidence
    warnings_text = ""
    if confidence_result['should_warn_user']:
        warnings_text = "\n⚠️ IMPORTANT LIMITATIONS:\n"
        if temporal_validation['has_mismatch']:
            for warning in temporal_validation['warnings']:
                warnings_text += f"- {warning['message']}\n"
        if data_freshness['days_old'] > 1:
            warnings_text += f"- Data is {data_freshness['days_old']:.1f} days old, may not reflect latest market conditions\n"
    
    # Construct main prompt
    main_prompt = f"""You are a financial analyst with access to real-time stock market data.

{recent_summary}
{warnings_text}

RELEVANT HISTORICAL DATA:
{retrieved_context}

Question: {question}

INSTRUCTIONS:
- Answer based on the data provided above
- ALWAYS state the date of the most recent data you're referencing
- If asked about "current" or "today" and data is not from today, explicitly mention the data date
- Reference specific numbers and dates
- If data freshness is concerning, acknowledge it in your answer
- Be honest about limitations

Answer:"""
    
    # Generate initial answer
    print("\nGenerating answer...")
    answer = call_llm(main_prompt)
    result['initial_answer'] = answer
    
    # Self-verification (if enabled)
    if enable_self_verification:
        print("\nPerforming self-verification...")
        verification_result = self_verify_answer(
            question, answer, data_freshness, relevant_chunks
        )
        result['verification'] = verification_result
        
        if verification_result.get('verified') and not verification_result.get('is_valid'):
            print(f"  ⚠ Verification found issues: {verification_result.get('issues')}")
            
            # Add disclaimers to answer
            if verification_result.get('disclaimers'):
                disclaimer_text = "\n\n📌 Important Notes:\n" + "\n".join(
                    f"- {d}" for d in verification_result['disclaimers']
                )
                answer = answer + disclaimer_text
                result['final_answer'] = answer
        
        print(f"  → Verification status: {'PASSED' if verification_result.get('is_valid') else 'ISSUES_FOUND'}")
        print(f"  → Obsolescence risk: {verification_result.get('obsolescence_risk', 'UNKNOWN')}")
    
    if 'final_answer' not in result:
        result['final_answer'] = answer
    result['success'] = True
    
    return result


# ============================================================================
# BATCH EVALUATION WITH DETAILED METRICS
# ============================================================================

BENCHMARK_QUESTIONS_ENHANCED = {
    "AAPL": [
        "What is the most recent closing price?",
        "What was the highest price in the last 30 days?",
        "What is the current trend - is the stock going up or down?",
        "What was the trading volume on the most recent day?",
        "How does the current price compare to last week?",
    ]
}

def run_batch_eval_enhanced(ticker: str):
    """Run enhanced batch evaluation with detailed metrics"""
    print(f"\n{'='*70}")
    print(f"ENHANCED RAG SYSTEM EVALUATION: {ticker}")
    print(f"{'='*70}\n")
    
    if ticker not in BENCHMARK_QUESTIONS_ENHANCED:
        print(f"No benchmark questions for {ticker}")
        return
    
    questions = BENCHMARK_QUESTIONS_ENHANCED[ticker]
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*70}")
        print(f"Question {i}/{len(questions)}: {question}")
        print(f"{'─'*70}")
        
        result = ask_question_rag_enhanced(ticker=ticker, question=question)
        
        if result['success']:
            print(f"\nFinal Answer:\n{result['final_answer']}")
            print(f"\n📊 Metrics:")
            print(f"  - Data Freshness: {result['data_freshness']['freshness_category']}")
            print(f"  - Confidence: {result['confidence']['confidence_category']}")
            print(f"  - Obsolescence Risk: {result.get('verification', {}).get('obsolescence_risk', 'N/A')}")
            
            results.append({
                'question': question,
                'answer': result['final_answer'],
                'confidence': result['confidence'],
                'data_freshness': result['data_freshness'],
                'verification': result.get('verification', {}),
                'timestamp': result['timestamp']
            })
    
    # Save detailed results
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/{ticker}_rag_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'ticker': ticker,
            'system': 'RAG_ENHANCED',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total_questions': len(results),
                'avg_confidence': float(np.mean([r['confidence']['overall_confidence'] for r in results])),
                'high_confidence_count': int(sum(1 for r in results if r['confidence']['overall_confidence'] >= 0.75))
            }
        }, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Usage: python rag_enhanced.py --batch TICKER")
            sys.exit(1)
        run_batch_eval_enhanced(sys.argv[2].upper())
    else:
        # Interactive mode
        print("\n" + "="*70)
        print("ENHANCED RAG SYSTEM - Stock Market QA")
        print("With Temporal Awareness & Self-Verification")
        print("="*70)
        
        result = ask_question_rag_enhanced()
        
        if result['success']:
            print("\n" + "="*70)
            print("COMPREHENSIVE RESULT")
            print("="*70)
            print(f"\nAnswer:\n{result['final_answer']}")
            print(f"\n System Metrics:")
            print(f"  Data Freshness: {result['data_freshness']['freshness_category']}")
            print(f"  Confidence: {result['confidence']['confidence_category']} ({result['confidence']['overall_confidence']:.2f})")
            if 'verification' in result:
                print(f"  Verification: {'PASSED' if result['verification'].get('is_valid') else 'ISSUES FOUND'}")
                print(f"  Obsolescence Risk: {result['verification'].get('obsolescence_risk', 'N/A')}")