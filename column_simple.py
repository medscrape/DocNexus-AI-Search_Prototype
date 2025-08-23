#!/usr/bin/env python3
"""
Simple GPT-Based Medical Code Resolver
======================================
A minimal 50-line agent that uses only GPT to resolve medical queries to codes.
"""

import json
import sys
import argparse
import os
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables (override existing ones)
load_dotenv(override=True)

# Configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def get_medical_codes(query: str) -> Dict[str, List[str]]:
    try:
        from openai import OpenAI
        
        # Check API key format
        if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format - should start with 'sk-'")
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        system_prompt = """You are a medical coding expert. Given a medical query, return ONLY valid JSON with medical codes.

Required JSON structure:
{
  "icd10": ["list", "of", "icd10", "codes"],
  "icd9": ["list", "of", "icd9", "codes"], 
  "cpt": ["list", "of", "cpt", "codes"],
  "hcpcs": ["list", "of", "hcpcs", "codes"],
  "loinc": ["list", "of", "loinc", "codes"],
  "snomed": ["list", "of", "snomed", "codes"]
}

Rules:
- Use current US medical coding standards
- Handle typos and spelling errors intelligently
- Return empty arrays [] if no relevant codes exist
- Be specific and accurate
- Include primary and common secondary codes
- Do NOT include NDC or RxNorm codes
- For procedure queries, focus heavily on CPT codes
- Include ALL relevant CPT codes for each procedure mentioned
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Find medical codes for: {query}"}
            ],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        
        # Get the raw response content
        raw_content = response.choices[0].message.content
        
        if not raw_content or raw_content.strip() == "":
            raise ValueError("Empty response from OpenAI API")
        
        try:
            raw_output = json.loads(raw_content)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}", file=sys.stderr)
            print(f"Content causing error: '{raw_content}'", file=sys.stderr)
            raise ValueError(f"Invalid JSON response from OpenAI API: {e}")

        # ✅ Normalize ICD codes → remove dots
        def strip_dots(codes: List[str]) -> List[str]:
            return [c.replace(".", "") for c in codes]

        raw_output["icd10"] = strip_dots(raw_output.get("icd10", []))
        raw_output["icd9"]  = strip_dots(raw_output.get("icd9", []))

        return raw_output
        
    except Exception as e:
        print(f"Error calling GPT: {e}", file=sys.stderr)
        print(f"Query that caused error: '{query}'", file=sys.stderr)
        return {
            "icd10": [], "icd9": [], "cpt": [], "hcpcs": [],
            "loinc": [], "snomed": []
        }

def main():
    parser = argparse.ArgumentParser(description="Simple GPT Medical Code Resolver")
    parser.add_argument("--query", "-q", required=True, help="Medical query to resolve")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")
    
    args = parser.parse_args()
    
    codes = get_medical_codes(args.query)
    
    if args.pretty:
        print(json.dumps(codes, indent=2))
    else:
        print(json.dumps(codes))

if __name__ == "__main__":
    main()
else:
    # When imported, don't run main automatically
    pass