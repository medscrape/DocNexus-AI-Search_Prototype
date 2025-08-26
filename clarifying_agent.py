#!/usr/bin/env python3
"""
Simple Query Clarifying Agent
============================
Takes queries, asks clarifying questions, returns improved query.
"""
import json
import argparse
import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

class QueryClarifyingAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

    def clarify_query(self, query, skip_confirmation=False):
        if skip_confirmation:
            return self._expand_query(query)
        
        # Ask clarifying questions
        questions = self._get_questions(query)
        
        if "QUERY_CLEAR" in questions:
            print("Query is clear, expanding...")
            return self._expand_query(query)
        
        # Get user clarification
        final_query = self._get_user_input(query, questions)
        return self._expand_query(final_query)

    def _get_questions(self, query):
        prompt = f"""You are a healthcare data analytics expert. Analyze this query for ambiguities that would significantly impact the analysis approach or results.

USER QUERY: "{query}"

Focus on identifying ambiguities in these critical areas:

DRUG/CONDITION SPECIFICITY:
- Drug class names could mean: specific drugs within that class or the entire therapeutic category
- Condition terms could mean: specific diagnoses or broader disease categories
- Brand vs generic names - which should be included
- Therapeutic categories vs individual medications

MEASUREMENT METRICS - What exactly to count/measure:
- "top" could mean: prescription volume, unique patient count, prescribing provider count etc.
- "popular" could mean: most prescriptions, most patients, most providers prescribing, highest growth rate  
- "volume" could mean: prescription count, dispensed units, days supply, total charges
- "usage" could mean: prescription utilization, patient adherence, dispensing frequency
- "performance" could mean: prescription volume, patient outcomes, adherence rates, cost metrics

AGGREGATION LEVEL - What entity to analyze:
- Individual providers vs hospitals vs health systems vs geographic regions
- Organization-level vs individual prescriber analysis  
- Patient-level vs population-level metrics

TIME SCOPE AMBIGUITIES:
- "recent" "current" "trending" "lately" - need specific timeframes
- "this year" "last year" - need exact year boundaries
- Missing time context entirely

HEALTHCARE DATA RELATIONSHIPS:
- Drug queries: prescription data vs provider specialty data vs referral patterns
- Provider queries: individual prescribers vs organizational affiliations vs referral networks
- Patient analysis: medical claims vs pharmacy claims vs combined analysis

GEOGRAPHIC/DEMOGRAPHIC SCOPE:
- State/regional focus vs national analysis
- Specific specialties vs all providers
- Patient demographics or payer types

Examples of good clarifying questions:
- "When you say 'top hospitals', do you want to rank by total prescription volume, number of unique patients, or something else?"
- "Are you looking for data from a specific time period, or all available data?"
- "Do you want individual provider results or hospital-level aggregations?"

Determine if the query needs clarification by checking if these ambiguities would lead to significantly different analyses.

If the query is sufficiently clear and specific, respond with: "QUERY_CLEAR"

If clarification is needed, write a brief, conversational response with 2-3 numbered questions focusing on the most critical ambiguities. Keep it clean and simple like this example:

"To better understand your request regarding [topic], could you clarify the following:
1. [First key question]
2. [Second key question]  
3. [Third key question if needed]"

Your response:"""

        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'gpt-4o-mini',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.3,
                    'max_tokens': 200
                }
            )
            return response.json()['choices'][0]['message']['content'].strip()
        except:
            return "QUERY_CLEAR"

    def _get_user_input(self, query, questions):
        print(f"\nQuery: \"{query}\"")
        print(f"Questions: {questions}")
        print("\nType clarification or 'proceed':")
        
        user_input = input().strip()
        
        if user_input.lower() == 'proceed':
            return query
        
        # Combine original + clarification
        combined_prompt = f"""Combine these into one clear query:
Original: "{query}"
Clarification: "{user_input}"

Return refined query:"""

        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'gpt-4o-mini',
                    'messages': [{'role': 'user', 'content': combined_prompt}],
                    'temperature': 0.1,
                    'max_tokens': 100
                }
            )
            refined = response.json()['choices'][0]['message']['content'].strip()
            return refined.strip('"').strip("'")
        except:
            return f"{query}. {user_input}"

    def _expand_query(self, query):
        prompt = f"""Expand this healthcare query by:
1. Clarifying vague terms ("top" → "highest prescription volume")
2. Standardizing medical terminology.

Query: "{query}"

Return expanded query:"""

        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'gpt-4o-mini',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.1,
                    'max_tokens': 150
                }
            )
            expanded = response.json()['choices'][0]['message']['content'].strip()
            return expanded.strip('"').strip("'")
        except:
            return query

def main():
    parser = argparse.ArgumentParser(description="Query Clarifying Agent")
    parser.add_argument("--query", "-q", required=True, help="Query to clarify")
    parser.add_argument("--skip-confirm", action="store_true", help="Skip confirmation")
    args = parser.parse_args()
    
    agent = QueryClarifyingAgent()
    final_query = agent.clarify_query(args.query, args.skip_confirm)
    
    print(f"\nFINAL QUERY: \"{final_query}\"")

if __name__ == "__main__":
    main()