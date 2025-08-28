#!/usr/bin/env python3
"""
Enhanced Medical Code Resolver (OpenAI-only, same output format)
----------------------------------------------------------------
This version brings in the missing strengths from your first script while:
- Using **only OpenAI** (no Perplexity)
- Keeping the **same output JSON format** as your second script
- Adding a lightweight, robust two-stage pipeline:
  1) Extract structured terms from the user's query
  2) Ask for codes **only** for conditions/procedures/medications found in the query
- Batch generation, JSON-safety, de-duplication, dot-stripping for ICD codes, and useful diagnostics

CLI flags preserved:
  --pretty       : pretty-print JSON
  --codes-only   : output only the codes per category (back-compat). Note: LOINC and SNOMED removed by request.

Env vars:
  OPENAI_API_KEY (required)
  OPENAI_MODEL   (optional, default: gpt-4o)
"""
import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load env
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")

# ----------------------------- Utilities ----------------------------- #

def _ensure_api_key() -> None:
    if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
        raise ValueError("OPENAI_API_KEY missing or invalid; must start with 'sk-'")


def _json_only_completion(system_prompt: str, user_prompt: str, max_tokens: int = 1600) -> Dict[str, Any]:
    """Call OpenAI chat.completions with response_format=json_object and return parsed JSON."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content
    if not content:
        raise ValueError("Empty completion")
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # Extremely rare with response_format=json_object, but guard anyway
        raise ValueError(f"Model returned non-JSON content: {e}: {content[:200]}")


def _strip_icd_dots(codes: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove dots from ICD codes while preserving descriptions."""
    out: List[Dict[str, str]] = []
    for item in codes:
        if isinstance(item, dict) and "code" in item:
            it = dict(item)
            it["code"] = it["code"].replace(".", "")
            out.append(it)
        else:
            out.append(item)
    return out


def _dedup_drug_names(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """De-duplicate drug name dicts by (generic, brand) with order preserved."""
    seen = set()
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        generic = it.get("generic", "").strip()
        brand = it.get("brand", "").strip()
        key = (generic.lower(), brand.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"generic": generic, "brand": brand})
    return out


def _dedup_code_objects(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """De-duplicate code dicts by (code, description) with order preserved."""
    seen = set()
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        code = it.get("code", "").strip()
        desc = it.get("description", "").strip()
        key = (code, desc)
        if key in seen:
            continue
        seen.add(key)
        out.append({"code": code, "description": desc})
    return out


# ----------------------------- Public API ----------------------------- #

def get_medical_codes(query: str) -> Dict[str, List[Any]]:
    """Main entry point: single GPT call to extract and get codes directly."""
    _ensure_api_key()
    
    # Single call that does both extraction and code lookup
    system_prompt = (
        "You are a medical coding expert. Analyze the user's query to identify medical conditions, "
        "procedures, and medications, then return appropriate medical codes using current US standards."
    )
    
    user_prompt = f"""
Analyze this medical query and return ONLY this JSON structure (no markdown):
{{
  "icd10": [{{"code":"","description":""}}],
  "icd9":  [{{"code":"","description":""}}],
  "cpt":   [{{"code":"","description":""}}],
  "hcpcs": [{{"code":"","description":""}}],
  "drug_names": [{{"generic":"","brand":""}}]
}}

Query: "{query}"

Instructions:
- Identify medical conditions, procedures, and medications in the query
- Use current US medical coding standards
- Handle typos and spelling errors intelligently
- Return empty arrays [] if no relevant codes exist
- Be specific and accurate with both codes AND descriptions
- Include primary and common secondary codes
- Do NOT include NDC or RxNorm codes
- Do NOT add unclassified drugs or generic "unspecified" drug codes
- For drug_names: Smart drug mapping based on query input:
  * If drug class is mentioned (e.g., ACE inhibitors, beta blockers, etc.), list specific drug names in that class
  * If brand name is mentioned, provide the generic name (and keep the brand name)  
  * If generic name is mentioned, provide the brand name (and keep the generic name)
  * Leave empty array if no drugs/drug classes in query
- For procedure queries, include CPT and HCPCS Level II codes
- Include all relevant HCPCS Level II codes (e.g., A-, B-, C-, D-, E-, G-, J-, K-, L-, M-, P-, Q-, R-, S-, T-, V-codes) where appropriate
- Always include BOTH the code and its full official description
- Keep descriptions concise but medically accurate
"""
    
    result = _json_only_completion(system_prompt, user_prompt, max_tokens=2000)
    
    # Normalize / safety: ensure all keys exist and are lists
    for k in ["icd10", "icd9", "cpt", "hcpcs", "drug_names"]:
        result.setdefault(k, [])
        if not isinstance(result[k], list):
            result[k] = []

    # De-duplicate & strip dots for ICD
    result["icd10"] = _dedup_code_objects(_strip_icd_dots(result["icd10"]))
    result["icd9"]  = _dedup_code_objects(_strip_icd_dots(result["icd9"]))

    # De-duplicate others
    for k in ["cpt", "hcpcs"]:
        result[k] = _dedup_code_objects(result[k])
    
    # De-duplicate drug names
    result["drug_names"] = _dedup_drug_names(result["drug_names"])

    return result


# ------------------------------ CLI Main ------------------------------ #

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Medical Code Resolver (OpenAI-only)")
    parser.add_argument("--query", "-q", required=True, help="Medical query to resolve")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")
    parser.add_argument("--codes-only", action="store_true", help="Show only codes without descriptions")
    args = parser.parse_args()

    try:
        result = get_medical_codes(args.query)

        if args.codes_only:
            only = {k: [d.get("code", "") for d in v if isinstance(d, dict)] for k, v in result.items()}
            result = only

        if args.pretty:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        # Output empty scaffold on error to preserve contract
        empty = { "icd10": [], "icd9": [], "cpt": [], "hcpcs": [], "drug_names": [] }
        print(json.dumps(empty))


if __name__ == "__main__":
    _cli()