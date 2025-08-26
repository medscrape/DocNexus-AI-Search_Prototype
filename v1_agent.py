#!/usr/bin/env python3
"""
NLP to ClickHouse Main Agent — with Auto Visualization Dimension
================================================================
- Does the original job (codes + tables → SQL → optional execution)
- **Also returns a visualization "dimension"** for your front end:
    - 1 → categorical breakdown (pie/bar)
    - 2 → time series (line/area)

Heuristic rules (fast, no extra API call):
- If the query/SQL indicates time-series (year/month/date, GROUP BY/ORDER BY over time), return **2**.
- Else if it looks like a single categorical grouping with an aggregate (e.g., `GROUP BY X` and `COUNT/SUM`), return **1**.
- Else default to **1** (safe default for pie/bar/table).

Returned JSON now contains a `viz` block like:
{
  "viz": { "dimension": 2, "chart": "line", "rationale": "Detected year/month grouping" }
}
"""

import json
import sys
import argparse
import os
import time
import re
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple
import requests
from dotenv import load_dotenv

# Load environment variables (override existing ones)
load_dotenv(override=True)

# Import our existing agents
from column_simple import get_medical_codes
from table_agent import MedicalTableIdentificationAgent
from clickhouse_agent import ClickHouseAgent

ChartInfo = Dict[str, Any]


def _contains_time_words(text: str) -> bool:
    text = text.lower()
    time_words = [
        "trend", "over time", "per month", "by month", "monthly",
        "per year", "by year", "yearly", "timeline", "time series",
        "moM", "YoY".lower()
    ]
    return any(w in text for w in time_words)


def _looks_time_grouped(sql: str) -> bool:
    s = sql.lower()
    # Obvious time cols / functions
    time_cols = [
        r"\byear\b", r"\bmonth\b", r"\bdate\b", r"\bservice_date_dd\b",
        r"\btransaction_dt\b", r"\bstatement_from_dd\b", r"\bstatement_to_dd\b",
        r"\bservice_from_dd\b", r"\bservice_to_dd\b",
        r"toyear\(", r"tomonth\(", r"toYYYYMM".lower(), r"toStartOfMonth".lower(), r"toDate".lower()
    ]
    if any(re.search(pat, s) for pat in time_cols):
        # If we also see grouping/order by over time → definitely time series
        if ("group by" in s and ("year" in s or "month" in s or "date" in s)) or \
           ("order by" in s and ("year" in s or "month" in s or "date" in s)):
            return True
    # Look for classic pattern "group by year, month" or similar
    if re.search(r"group\s+by\s+.*\byear\b.*\bmonth\b", s):
        return True
    return False


def _extract_group_by_columns(sql: str) -> List[str]:
    """Extract GROUP BY columns from SQL query."""
    s = re.sub(r"\s+", " ", sql, flags=re.MULTILINE)
    m = re.search(r"group\s+by\s+(.+?)(?:\s+order\s+by|\s+limit|\s+format|$)", s, flags=re.IGNORECASE)
    if not m:
        return []
    
    cols = m.group(1)
    # Simple split by comma (ignoring nested functions for now)
    return [c.strip() for c in cols.split(",") if c.strip()]


def _has_clear_aggregate(sql: str) -> bool:
    """Check if SQL has aggregation functions."""
    s = sql.lower()
    return any(fn in s for fn in ["count(", "sum(", "avg(", "uniq(", "min(", "max("])



def infer_viz_dimension(original_query: str, sql: str) -> ChartInfo:
    """
    Infer visualization dimension and chart type based on query and SQL analysis.
    
    Returns:
    - dimension 1 + "pie": For simple categorical breakdowns or single values
    - dimension 2 + "bar": For categorical comparisons, rankings, top N lists
    - dimension 2 + "line": For time series data
    """
    
    query_lower = original_query.lower()
    sql_lower = sql.lower()
    
    # === SINGLE VALUE DETECTION (dimension 1) - Check this FIRST ===
    
    # Check if SQL has no GROUP BY - this means single aggregate result
    has_group_by = "group by" in sql_lower
    has_sql_agg = any(func in sql_lower for func in ["count(", "sum(", "avg(", "uniq(", "min(", "max("])
    
    # If aggregation without grouping = single value result
    if has_sql_agg and not has_group_by:
        return {
            "dimension": 1,
            "chart": "pie",  # Or could be "metric" for single number display
            "rationale": "Single aggregate value - no grouping detected in SQL"
        }
    
    # === TIME SERIES DETECTION (dimension 2, line chart) ===
    
    # 1. Explicit time-series keywords
    time_keywords = [
        "trend", "over time", "per month", "by month", "monthly",
        "per year", "by year", "yearly", "annually", "timeline", 
        "time series", "mom", "yoy", "quarter", "weekly", "daily"
    ]
    if any(keyword in query_lower for keyword in time_keywords):
        return {
            "dimension": 2,
            "chart": "line",
            "rationale": f"Detected time-series keywords: {[k for k in time_keywords if k in query_lower]}"
        }
    
    # 2. Year range pattern (e.g., "2020 to 2024", "between 2019 and 2023")
    year_range_patterns = [
        r"\b(19|20)\d{2}\s*(to|through|-)\s*(19|20)\d{2}\b",
        r"between\s+(19|20)\d{2}\s+and\s+(19|20)\d{2}\b",
        r"from\s+(19|20)\d{2}\s+(to|through|until)\s+(19|20)\d{2}\b"
    ]
    for pattern in year_range_patterns:
        if re.search(pattern, query_lower):
            # If SQL also groups by time, definitely time series
            if re.search(r"group\s+by\s+.*(year|month|date)", sql_lower):
                return {
                    "dimension": 2,
                    "chart": "line",
                    "rationale": "Year range detected with GROUP BY on time column"
                }
    
    # 3. SQL has GROUP BY with time columns
    time_sql_patterns = [
        r"group\s+by\s+.*\b(year|month|date)\b",
        r"group\s+by\s+.*to(year|month|date)\(",
        r"group\s+by\s+.*toyyyy",
        r"order\s+by\s+.*\b(year|month|date)\b"
    ]
    if any(re.search(pattern, sql_lower) for pattern in time_sql_patterns):
        return {
            "dimension": 2,
            "chart": "line",
            "rationale": "SQL groups/orders by time columns"
        }
    
    # === BAR CHART DETECTION (dimension 2, bar chart) ===
    # Only proceed if we have GROUP BY (multiple categories to compare)
    
    if not has_group_by:
        # No grouping = not suitable for bar chart
        return {
            "dimension": 1,
            "chart": "pie",
            "rationale": "No grouping found - single value or simple result"
        }
    
    # 1. Ranking/comparison keywords
    ranking_keywords = [
        "top", "bottom", "highest", "lowest", "most", "least",
        "best", "worst", "rank", "compare", "comparison",
        "leading", "popular", "common", "frequent", "volume",
        "largest", "smallest", "maximum", "minimum"
    ]
    has_ranking = any(keyword in query_lower for keyword in ranking_keywords)
    
    # 2. Count/aggregate keywords (only relevant if we have grouping)
    aggregate_keywords = [
        "count", "total", "sum", "volume", "number of",
        "how many", "quantity", "amount"
    ]
    has_aggregate = any(keyword in query_lower for keyword in aggregate_keywords)
    
    # 3. "By" grouping keywords (e.g., "by state", "by provider", "by drug")
    by_grouping = re.search(r"\bby\s+\w+", query_lower) is not None
    
    # 4. LIMIT clause suggests top-N query
    has_limit = "limit" in sql_lower
    
    # 5. ORDER BY suggests ranking
    has_order_by = "order by" in sql_lower
    
    # Bar chart scoring (only if we have GROUP BY)
    bar_score = 0
    reasons = []
    
    if has_ranking:
        bar_score += 3
        reasons.append("ranking keywords")
    if has_aggregate and has_group_by:  # Only count if grouped
        bar_score += 2
        reasons.append("aggregation with grouping")
    if by_grouping:
        bar_score += 2
        reasons.append("'by' grouping pattern")
    if has_group_by:
        bar_score += 2
        reasons.append("SQL GROUP BY")
    if has_sql_agg and has_group_by:  # Only count if grouped
        bar_score += 2
        reasons.append("SQL aggregation with grouping")
    if has_limit:
        bar_score += 1
        reasons.append("SQL LIMIT")
    if has_order_by:
        bar_score += 1
        reasons.append("SQL ORDER BY")
    
    # If strong bar chart signals, return bar chart
    if bar_score >= 4:
        return {
            "dimension": 2,
            "chart": "bar",
            "rationale": f"Bar chart indicators (score: {bar_score}): {', '.join(reasons)}"
        }
    
    # === PIE CHART DETECTION (dimension 1, pie chart) ===
    
    # Pie charts work best for:
    # - Simple categorical breakdowns
    # - "Distribution of", "breakdown by", "split by"
    # - Single GROUP BY with reasonable number of categories
    
    pie_keywords = [
        "distribution", "breakdown", "split", "share", "percentage",
        "proportion", "composition", "makeup"
    ]
    has_pie_keywords = any(keyword in query_lower for keyword in pie_keywords)
    
    # Check if it's a simple single-column GROUP BY (good for pie)
    group_by_columns = _extract_group_by_columns(sql)
    is_simple_grouping = len(group_by_columns) == 1
    
    # Pie chart conditions
    if has_pie_keywords and has_group_by:
        return {
            "dimension": 1,
            "chart": "pie",
            "rationale": "Distribution/breakdown keywords with grouping"
        }
    
    if is_simple_grouping and has_sql_agg and not (has_ranking or has_limit or has_order_by):
        return {
            "dimension": 1,
            "chart": "pie",
            "rationale": f"Simple categorical grouping: {group_by_columns[0]}"
        }
    
    # === DEFAULT LOGIC ===
    
    # If we have grouping but low bar score, could still be pie-worthy
    if has_group_by and bar_score < 4:
        return {
            "dimension": 1,
            "chart": "pie",
            "rationale": "Categorical grouping, not strong enough for bar chart"
        }
    
    # If we have aggregation + grouping but no strong signals
    if has_group_by and has_sql_agg:
        return {
            "dimension": 2,
            "chart": "bar",
            "rationale": "Grouped aggregation - defaulting to bar chart"
        }
    
    # Final fallback
    return {
        "dimension": 1,
        "chart": "pie",
        "rationale": "No strong signals - defaulting to simple display"
    }

class NLPToClickHouseAgent:
    """Main coordinator agent that generates and executes ClickHouse queries from NLP"""

    def __init__(self, api_key: Optional[str] = None):
        # Initialize sub-agents
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        self.table_agent = MedicalTableIdentificationAgent(self.api_key)
        self.clickhouse_agent = ClickHouseAgent()  # Initialize ClickHouse agent

        # Time tracking
        self.timing_data = {}

        # Hardcoded table schemas based on actual CSV files
        self.table_schemas = {
            'medical_blue': {
                'columns': [
                    'year', 'month', 'patient_gender', 'patient_age', 'procedure_code',
                    'diagnosis_code', 'type_1_npi', 'type_2_npi', 'patient_id', 'claim_number',
                    'claim_category', 'claim_charge_amount', 'diagnosis_vocabulary_id',
                    'diagnosis_description', 'procedure_vocabulary_id', 'procedure_description',
                    'ndc', 'date', 'payer_name', 'payer_channel_name', 'payer_subchannel_name',
                    'final_status_code', 'admitting_diagnosis_vocabulary_id', 'assigned_ms_drg',
                    'claim_admission_type_cd', 'claim_drg_cd', 'claim_line_charge_amt',
                    'claim_line_seq_num', 'claim_submitted_dd', 'claim_type_cd', 'closed_source_fl',
                    'days_or_units_val', 'encounter_channel_nm', 'encounter_subchannel_nm',
                    'header_anchor_dd', 'open_source_fl', 'pos_cd', 'primary_diagnosis_cd',
                    'primary_diagnosis_description', 'primary_diagnosis_vocabulary_id',
                    'primary_hco_name', 'primary_hco_provider_classification', 'primary_hco_source',
                    'primary_hcp_name', 'primary_hcp_segment', 'primary_hcp_source',
                    'principal_procedure_cd', 'principal_procedure_vocabulary_id',
                    'procedure_modifier_1_cd', 'procedure_modifier_1_desc', 'referring_npi_nbr',
                    'referring_npi_nm', 'rendering_provider_classtype', 'rendering_provider_segment',
                    'rendering_provider_state', 'rendering_provider_zip', 'resubmitted_claim_nbr',
                    'revenue_cd', 'service_from_dd', 'service_to_dd', 'statement_from_dd',
                    'statement_to_dd', 'source_file_key'
                ],
                'key_columns': {
                    'patient_id': 'Unique patient identifier (UUID)',
                    'diagnosis_code': 'Medical diagnosis code (ICD-9/ICD-10)',
                    'procedure_code': 'Medical procedure code (CPT/HCPCS)',
                    'primary_diagnosis_cd': 'Primary diagnosis code for encounter',
                    'principal_procedure_cd': 'Principal procedure code for encounter',
                    'type_1_npi': 'Healthcare provider NPI number',
                    'type_2_npi': 'Healthcare organization NPI number',
                    'referring_npi_nbr': 'NPI number of referring provider',
                    'referring_npi_nm': 'Name of referring provider',
                    'primary_hcp_name': 'Name of primary healthcare provider',
                    'primary_hco_name': 'Name of primary healthcare organization',
                    'service_from_dd': 'Start date of service period',
                    'service_to_dd': 'End date of service period',
                    'year': 'Year of medical claim',
                    'month': 'Month of medical claim'
                }
            },
            'pharmacy_blue': {
                'columns': [
                    'year', 'month', 'patient_gender', 'patient_age', 'drug_generic_name',
                    'drug_brand_name', 'type_1_npi', 'type_2_npi', 'patient_id', 'date',
                    'claim_number', 'transaction_status', 'ndc', 'payer_name', 'payer_channel_name',
                    'payer_subchannel_name', 'final_status_code', 'service_date_dd',
                    'date_prescription_written_dd', 'transaction_dt', 'dispense_nbr',
                    'admin_service_line', 'clinical_service_line', 'reject_reason_1_cd',
                    'reject_reason_1_desc', 'open_source_fl', 'closed_source_fl', 'ndc_desc',
                    'ndc_drug_nm', 'ndc_isbranded_ind', 'roa', 'prescriber_npi_state_cd',
                    'prescriber_npi_nm', 'pharmacy_npi_nbr', 'pcp_npi_nbr', 'payer_id',
                    'payer_company_nm', 'payer_bin_nbr', 'days_supply_val', 'awp_unit_price_amt',
                    'total_paid_amt', 'patient_to_pay_amt', 'update_ts', 'source_file_key'
                ],
                'key_columns': {
                    'patient_id': 'Unique patient identifier (UUID)',
                    'drug_brand_name': 'Brand name of medication (use for Enspryng)',
                    'drug_generic_name': 'Generic name of medication (use for satralizumab)',
                    'type_1_npi': 'Healthcare provider NPI number',
                    'type_2_npi': 'Healthcare organization NPI number',
                    'prescriber_npi_nm': 'Name associated with prescriber NPI',
                    'pharmacy_npi_nbr': 'Pharmacy NPI number',
                    'service_date_dd': 'Date when service was provided',
                    'year': 'Year of prescription',
                    'month': 'Month of prescription'
                }
            },
            'as_providers_v1': {
                'columns': [
                    'type_1_npi', 'type_2_npi_names', 'type_2_npis', 'first_name', 'middle_name',
                    'last_name', 'gender', 'specialties', 'conditions_tags', 'conditions',
                    'cities', 'states', 'counties', 'city_states', 'hospital_names',
                    'system_names', 'affiliations', 'best_type_2_npi', 'best_hospital_name',
                    'best_system_name', 'phone', 'email', 'linkedin', 'twitter',
                    'has_youtube', 'has_podcast', 'has_linkedin', 'has_twitter',
                    'num_payments', 'num_clinical_trials', 'num_publications'
                ],
                'key_columns': {
                    'type_1_npi': 'Healthcare provider NPI number (individual)',
                    'type_2_npi_names': 'Names of healthcare organizations (Array)',
                    'type_2_npis': 'Organization NPI numbers (Array)',
                    'first_name': 'Provider first name',
                    'last_name': 'Provider last name',
                    'gender': 'Provider gender',
                    'specialties': 'Provider specialties (likely Array)',
                    'conditions_tags': 'Condition tags provider treats (likely Array)',
                    'conditions': 'Medical conditions provider treats (likely Array)',
                    'cities': 'Cities where provider practices (likely Array)',
                    'states': 'States where provider practices (likely Array)',
                    'hospital_names': 'Hospital affiliations (likely Array)',
                    'system_names': 'Health system affiliations (likely Array)',
                    'best_hospital_name': 'Primary hospital affiliation (String)',
                    'best_system_name': 'Primary health system affiliation (String)'
                }
            }
        }

    def _build_comprehensive_prompt(self, query: str, medical_codes: Dict, required_tables: List[str]) -> str:
        """Build comprehensive prompt for ClickHouse query generation"""

        prompt = f"""You are an expert ClickHouse SQL developer specializing in medical data analytics.

TASK: Convert the natural language query into a precise ClickHouse SQL query.

USER QUERY: "{query}"

EXTRACTED MEDICAL CODES:
{json.dumps(medical_codes, indent=2)}

REQUIRED TABLES: {', '.join(required_tables)}

TABLE SCHEMAS AND DESCRIPTIONS:
"""

        # Add detailed table information
        for table_name in required_tables:
            if table_name in self.table_agent.table_catalog:
                table_info = self.table_agent.table_catalog[table_name]
                prompt += f"\n--- TABLE: {table_name} ---\n"
                prompt += f"Purpose: {table_info['purpose']}\n"
                prompt += f"Core Data: {table_info['core_data']}\n"
                prompt += "Capabilities:\n"
                for cap in table_info['analytical_capabilities']:
                    prompt += f"  - {cap}\n"

                # Add actual schema from hardcoded schemas
                if table_name in self.table_schemas:
                    schema = self.table_schemas[table_name]
                    prompt += f"\nAll Columns: {', '.join(schema['columns'])}\n"
                    prompt += "\nKey Columns for Queries:\n"
                    for col, desc in schema['key_columns'].items():
                        prompt += f"  - {col}: {desc}\n"

        prompt += f"""

SQL GENERATION RULES:
1. Generate ONLY the ClickHouse SELECT statement - no connection code
2. Use proper ClickHouse syntax (not MySQL/PostgreSQL)
3. Apply extracted medical codes as WHERE conditions when relevant
4. Use ClickHouse functions like count(), uniq(), groupArray() appropriately
5. Handle multiple tables with appropriate JOINs if needed
6. Use EXACT column names from the schemas above
7. Apply date filtering using ClickHouse date functions when temporal analysis is needed
8. For aggregations, use ClickHouse-specific functions like uniqCombined64()
9. Use FORMAT for output formatting if helpful (e.g., FORMAT JSON, FORMAT TabSeparated)
10. Include appropriate ORDER BY and LIMIT clauses for result management
11. Use Pattern matching like "diagnosis_code ILIKE '%C34%' for medical conditions"
12. - For drug class queries, search by specific drug names not class names - drug classes are not literal database values

MEDICAL CODE MATCHING RULES:
- For ICD codes: Use 'diagnosis_code' column in medical_blue table ONLY
- For procedures/HCPCS: Use 'procedure_code' column in medical_blue table ONLY
- For drugs by name: Use 'drug_brand_name' or 'drug_generic_name' columns in pharmacy_blue table ONLY
- Patient linking: Use 'patient_id' column (exists in both medical_blue and pharmacy_blue)

COLUMN RESTRICTIONS:
- pharmacy_blue table does NOT have: rxnorm_code, procedure_code, hcpcs_code, diagnosis_code
- medical_blue table does NOT have: drug_brand_name, drug_generic_name
- For Enspryng medication: Use pharmacy_blue.drug_brand_name = 'Enspryng' OR pharmacy_blue.drug_generic_name = 'satralizumab'
- For HCPCS codes like J3490: Use medical_blue.procedure_code = 'J3490' (NOT pharmacy_blue)

SPECIFIC COLUMN USAGE:
medical_blue table:
- patient_id: for patient identification
- diagnosis_code: for ICD-9/ICD-10 codes (like 'G360' for NMOSD)
- procedure_code: for CPT/HCPCS codes
- provider_npi: for provider identification
- year, month: for temporal filtering

pharmacy_blue table:
- patient_id: for patient identification (UUID)
- drug_brand_name: for brand names (like 'Enspryng')
- drug_generic_name: for generic names (like 'satralizumab')
- type_1_npi: healthcare provider NPI number
- type_2_npi: healthcare organization NPI number
- prescriber_npi_nm: name associated with prescriber NPI
- pharmacy_npi_nbr: pharmacy NPI number
- service_date_dd: date when service was provided
- year, month: for temporal filtering
- DOES NOT HAVE: rxnorm_code, procedure_code, hcpcs_code, diagnosis_code

as_providers_v1 table:
- type_1_npi: Healthcare provider NPI number (individual) - use for JOINs
- first_name, last_name: Provider names (String columns)
- gender: Provider gender (String)
- specialties: Provider specialties (Array) - use has() or arrayExists()
- conditions: Medical conditions provider treats (Array) - use has() or arrayExists()  
- cities: Cities where provider practices (Array) - use has() or arrayExists()
- states: States where provider practices (Array) - use has() or arrayExists()
- hospital_names: Hospital affiliations (Array) - use has() or arrayExists()
- system_names: Health system affiliations (Array) - use has() or arrayExists()
- best_hospital_name: Primary hospital affiliation (String)
- best_system_name: Primary health system affiliation (String)

ARRAY COLUMN HANDLING:
- Many columns in as_providers_v1 are Arrays: states, cities, specialties, conditions, hospital_names, etc.
- Use has() function to check if array contains a value: has(array_column, 'value')
- Use hasAny() to check multiple values: hasAny(array_column, ['value1', 'value2'])
- Use arrayExists() for pattern matching: arrayExists(x -> x ILIKE '%pattern%', array_column)
- Examples:
  * has(states, 'CALIFORNIA') - check if array contains 'CALIFORNIA'
  * arrayExists(x -> x ILIKE '%NEURO%', specialties) - pattern match in array
  * has(cities, 'NEW YORK') - check if provider practices in New York
  * arrayExists(x -> x ILIKE '%HOSPITAL%', hospital_names) - find hospitals with 'hospital' in name

DATA CONTEXT:

- All data is from USA by default (no need to filter by country)
- If query asks for "top in USA" or "doctors in the US" - you don't have to do anything at all as data is already from USA
- State names in arrays are in ALL CAPS format (e.g., 'CALIFORNIA', 'NEW YORK', 'TEXAS')
- Use ILIKE for case-insensitive pattern matching on state names
- Examples:
-

- has(states, 'CALIFORNIA') - exact state match
- arrayExists(x -> x ILIKE '%CALIF%', states) - pattern match for California
- has(states, 'NEW YORK') - exact match for New York state

NPI COLUMN DATA TYPES:
- type_1_npi and type_2_npi columns may be numeric (UInt64) not strings
- For numeric NPI columns, use IS NOT NULL instead of != '' for empty checks
- Examples:
  * CORRECT: WHERE type_2_npi IS NOT NULL AND type_2_npi > 0
  * WRONG: WHERE type_2_npi IS NOT NULL AND type_2_npi != ''
- Use > 0 to exclude zero values which may represent missing NPIs


NPI-BASED ORGANIZATION QUERIES:
- For "top organizations" queries, use type_2_npi directly from the main table (pharmacy_blue or medical_blue)
- type_2_npi represents the organization/facility NPI number
- This is simpler and more accurate than complex JOINs with provider tables
- Examples:
  * SELECT type_2_npi, count(*) FROM pharmacy_blue GROUP BY type_2_npi
  * SELECT type_2_npi, count(*) FROM medical_blue GROUP BY type_2_npi
- Only JOIN with as_providers_v1 if you specifically need organization names
- For organization names, use best_hospital_name or best_system_name (String columns, not arrays)

NULL/EMPTY VALUE FILTERING:
- Filter out NULL and empty values from results using WHERE clauses
- Use IS NOT NULL AND column != '' to exclude empty values
- For rankings/top lists: Always filter out NULLs/empties before counting
- Examples:
  * WHERE hospital_name IS NOT NULL AND hospital_name != ''
  * WHERE prescriber_npi_nm IS NOT NULL AND prescriber_npi_nm != ''
  * WHERE drug_brand_name IS NOT NULL AND drug_brand_name != ''
  * WHERE nullIf(trim(column), '') IS NOT NULL  -- handles whitespace too

US DATA FILTERING:
- ALL data is already from USA - NEVER filter by country
- NEVER use has(states, 'US') or similar country filters
- State arrays contain state names like 'CALIFORNIA', 'NEW YORK', 'TEXAS' - NOT country codes
- For US queries, simply ensure arrayLength(states) > 0 if you need state validation
- Examples of WRONG usage: has(states, 'US'), has(states, 'USA'), has(states, 'UNITED STATES')
- Examples of CORRECT usage: arrayLength(states) > 0, has(states, 'CALIFORNIA')

QUERY PATTERNS:
- Counting: SELECT count(*) FROM table WHERE condition
- Patient counts: SELECT count(DISTINCT patient_id) FROM table WHERE condition
- Aggregation: SELECT column, count(*) FROM table WHERE column IS NOT NULL AND column != '' GROUP BY column
- Top N: SELECT column, count(*) as cnt FROM table WHERE column IS NOT NULL AND column != '' GROUP BY column ORDER BY cnt DESC LIMIT 10
- Trends: SELECT year, month, count(*) FROM table GROUP BY year, month ORDER BY year, month
- Patient overlap: Use JOINs on patient_id between medical_blue and pharmacy_blue
- Always filter NULLs: Add WHERE column IS NOT NULL AND column != '' for cleaner results
- Array filtering: Use has() or arrayExists() for Array columns

FILTERING RULES:
- Always exclude NULL and empty string values from GROUP BY columns
- Use WHERE column IS NOT NULL AND column != '' before GROUP BY
- For text columns that might have whitespace: WHERE nullIf(trim(column), '') IS NOT NULL
- For Array columns: Use arrayLength(array_column) > 0 to ensure non-empty arrays
- This ensures top N results only show actual values, not empty entries

CRITICAL: Use ONLY the exact column names listed in the schemas above.

COLUMNS THAT DO NOT EXIST:
- pharmacy_blue does NOT have: rxnorm_code, procedure_code, hcpcs_code, diagnosis_code
- medical_blue does NOT have: drug_brand_name, drug_generic_name, rxnorm_code

CORRECT USAGE EXAMPLES:
- Patient linking: m.patient_id = p.patient_id (both are UUID type)
- State filtering: has(states, 'CALIFORNIA') or arrayExists(x -> x ILIKE '%CALIF%', states)
- Array pattern matching: arrayExists(x -> x ILIKE '%CARDIO%', specialties)
- Non-empty arrays: WHERE arrayLength(states) > 0
- Use pattern matching with ILIKE '%drug_name%' for more reliable matches

HOSPITAL NAME MATCHING:
- Hospital names may not match exactly as mentioned in queries
- Use ILIKE pattern matching for hospital names to handle variations

Do not use column names like 'icd10_code', 'hcpcs_code', 'ndc_code', 'rxnorm_code' - these do not exist.

PERFORMANCE OPTIMIZATION RULES:
- Filter large tables BEFORE joining to reduce dataset size
- For drug-diagnosis queries, filter pharmacy_blue first, then join
- Use subqueries or CTEs to pre-filter when dealing with large tables
- Limit timeframes when possible (specific months, not full years)
- Examples:
  * WRONG: Large JOIN then filter
  * CORRECT: Filter first, then smaller JOIN


Return ONLY the ClickHouse SQL query, nothing else.
"""

        return prompt

    def _run_parallel_analysis(self, query: str) -> tuple:
        """Run medical code extraction and table identification in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Check if medical codes are already provided (from ExecuteWithCodes endpoint)
            if hasattr(self, '_provided_medical_codes') and self._provided_medical_codes:
                # Use provided codes, only run table identification
                medical_codes = self._provided_medical_codes
                required_tables = self.table_agent.identify_tables(query)
                # Clean up the temporary attribute
                delattr(self, '_provided_medical_codes')
            else:
                # Submit both tasks simultaneously (original behavior)
                codes_future = executor.submit(get_medical_codes, query)
                tables_future = executor.submit(self.table_agent.identify_tables, query)

                # Wait for both to complete
                medical_codes = codes_future.result()
                required_tables = tables_future.result()

            return medical_codes, required_tables

    def generate_and_execute_query(self, query: str, execute: bool = True) -> Dict[str, Any]:
        """Main method to generate and optionally execute ClickHouse query from natural language"""

        # Start overall timing
        start_time = time.time()
        self.timing_data = {'stages': {}}

        try:
            # Step 1: Run parallel analysis (medical codes + table identification)
            print("🔍 Extracting medical codes and identifying tables in parallel...")
            parallel_start = time.time()

            medical_codes, required_tables = self._run_parallel_analysis(query)

            parallel_time = time.time() - parallel_start
            self.timing_data['stages']['parallel_analysis'] = round(parallel_time * 1000, 2)
            print(f"✅ Parallel analysis completed in {parallel_time:.3f}s")

            # Step 2: Generate ClickHouse query using GPT
            print("⚙️ Generating ClickHouse query...")
            gpt_start = time.time()

            prompt = self._build_comprehensive_prompt(query, medical_codes, required_tables)

            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'gpt-4o',  # Using GPT-4o as specified
                    'messages': [
                        {
                            'role': 'system',
                            'content': 'You are a ClickHouse SQL expert. Return only clean SQL queries without markdown or explanations.'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    'temperature': 0.1,
                    'max_tokens': 1000
                }
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")

            result = response.json()
            clickhouse_query = result['choices'][0]['message']['content'].strip()

            # Clean up the query (remove markdown if present)
            if clickhouse_query.startswith('```sql'):
                clickhouse_query = clickhouse_query.replace('```sql', '').replace('```', '').strip()
            elif clickhouse_query.startswith('```'):
                clickhouse_query = clickhouse_query.replace('```', '').strip()

            gpt_time = time.time() - gpt_start
            self.timing_data['stages']['gpt_generation'] = round(gpt_time * 1000, 2)
            print(f"✅ Query generated in {gpt_time:.3f}s")

            # 🧠 New: Infer visualization dimension (1=pie/bar, 2=line)
            viz = infer_viz_dimension(query, clickhouse_query)

            # Step 3: Execute the query if requested
            execution_result = None
            if execute:
                print("🚀 Executing ClickHouse query...")
                exec_start = time.time()

                execution_result = self.clickhouse_agent.execute(clickhouse_query)

                exec_time = time.time() - exec_start
                self.timing_data['stages']['query_execution'] = round(exec_time * 1000, 2)
                print(f"✅ Query executed in {exec_time:.3f}s")

            # Calculate total time
            total_time = time.time() - start_time
            self.timing_data['total_time_ms'] = round(total_time * 1000, 2)
            self.timing_data['total_time_s'] = round(total_time, 3)

            return {
                'success': True,
                'query': clickhouse_query,
                'execution_result': execution_result,  # ← DB result
                'medical_codes': medical_codes,
                'required_tables': required_tables,
                'final_prompt': prompt,  # Store the prompt for display
                'timing': self.timing_data,
                'viz': viz,              # ← New block with dimension/chart/rationale
                'metadata': {
                    'original_query': query,
                    'tables_used': required_tables,
                    'codes_extracted': sum(len(codes) for codes in medical_codes.values()),
                    'executed': execute
                }
            }

        except Exception as e:
            total_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'query': None,
                'execution_result': None,
                'medical_codes': medical_codes if 'medical_codes' in locals() else {},
                'required_tables': required_tables if 'required_tables' in locals() else [],
                'timing': {
                    'total_time_ms': round(total_time * 1000, 2),
                    'total_time_s': round(total_time, 3),
                    'stages': getattr(self, 'timing_data', {}).get('stages', {}),
                    'error_at_stage': self._determine_error_stage(e)
                },
                'viz': {"dimension": 1, "chart": "pie", "rationale": "Default on error."}
            }

    def generate_clickhouse_query(self, query: str) -> Dict[str, Any]:
        """Legacy method for backward compatibility - generates query without execution"""
        result = self.generate_and_execute_query(query, execute=False)
        # Ensure viz exists even when not executing
        if result.get('success') and 'viz' not in result:
            result['viz'] = infer_viz_dimension(query, result.get('query', ''))
        return result

    def execute_query(self, clickhouse_query: str) -> Dict[str, Any]:
        """Execute a ClickHouse query directly"""
        try:
            start_time = time.time()
            result = self.clickhouse_agent.execute(clickhouse_query)
            execution_time = time.time() - start_time

            # Even for direct execution, attempt a viz guess from SQL
            viz = infer_viz_dimension("", clickhouse_query)

            return {
                'success': result is not None,
                'execution_result': result,
                'execution_time_ms': round(execution_time * 1000, 2),
                'query': clickhouse_query,
                'viz': viz
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_result': None,
                'query': clickhouse_query,
                'viz': {"dimension": 1, "chart": "pie", "rationale": "Default on error."}
            }

    def _determine_error_stage(self, error: Exception) -> str:
        """Determine which stage the error occurred in"""
        error_str = str(error).lower()
        if 'openai' in error_str or 'api' in error_str:
            return 'gpt_generation'
        elif 'clickhouse' in error_str or 'database' in error_str:
            return 'query_execution'
        elif 'medical' in error_str or 'code' in error_str:
            return 'parallel_analysis'
        else:
            return 'unknown'

    def get_timing_summary(self) -> str:
        """Get a formatted timing summary"""
        if not hasattr(self, 'timing_data') or not self.timing_data:
            return "No timing data available"

        summary = []
        summary.append(f"📊 PERFORMANCE SUMMARY")
        summary.append(f"{'='*50}")

        # Calculate and show individual stage times
        stages = self.timing_data.get('stages', {})
        total_stages_ms = 0

        if stages:
            summary.append("Stage Breakdown:")
            for stage, time_ms in stages.items():
                stage_name = stage.replace('_', ' ').title()
                summary.append(f"  • {stage_name}: {time_ms}ms")
                total_stages_ms += time_ms

        # Show total from stages calculation
        total_stages_s = total_stages_ms / 1000
        summary.append(f"\n📈 TOTAL TIME (sum of stages): {total_stages_s:.3f}s ({total_stages_ms:.2f}ms)")

        # Also show the overall measured time for comparison
        overall_time_s = self.timing_data.get('total_time_s', 0)
        overall_time_ms = self.timing_data.get('total_time_ms', 0)
        summary.append(f"📈 TOTAL TIME (measured): {overall_time_s}s ({overall_time_ms}ms)")

        return "\n".join(summary)

    def write_timing_to_file(self, query: str, filename: str = "output.txt") -> None:
        """Write timing information to file"""
        import datetime

        try:
            with open(filename, 'a', encoding='utf-8') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Calculate total from stages
                stages = self.timing_data.get('stages', {})
                total_stages_ms = sum(stages.values())
                total_stages_s = total_stages_ms / 1000

                # Overall measured time
                overall_time_s = self.timing_data.get('total_time_s', 0)

                f.write(f"\n{timestamp} | Query: {query[:60]}{'...' if len(query) > 60 else ''}\n")
                f.write(f"  Total Time (stages): {total_stages_s:.3f}s ({total_stages_ms:.2f}ms)\n")
                f.write(f"  Total Time (measured): {overall_time_s:.3f}s\n")

                if stages:
                    f.write(f"  Stage Details: ")
                    stage_details = []
                    for stage, time_ms in stages.items():
                        stage_details.append(f"{stage}={time_ms}ms")
                    f.write(", ".join(stage_details) + "\n")

                f.write("-" * 80 + "\n")

        except Exception as e:
            print(f"⚠️  Warning: Could not write to {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description="NLP to ClickHouse Query Generator and Executor")
    parser.add_argument("--query", "-q", required=True, help="Natural language query")
    parser.add_argument("--pretty", "-p", action="store_true", help="Pretty print JSON output")
    parser.add_argument("--debug", "-d", action="store_true", help="Show debug information")
    parser.add_argument("--no-execute", action="store_true", help="Generate query only, don't execute")
    parser.add_argument("--timing", "-t", action="store_true", help="Show detailed timing information")

    args = parser.parse_args()

    # Initialize the main agent
    print("🚀 Initializing NLP to ClickHouse Agent...")
    agent = NLPToClickHouseAgent()

    # Generate and optionally execute the ClickHouse query
    execute_query = not args.no_execute
    result = agent.generate_and_execute_query(args.query, execute=execute_query)

    if result['success']:
        print(f"\n✅ Pipeline {'Completed' if execute_query else 'Generated Query'} Successfully!")

        # Show timing summary
        if args.timing or args.debug:
            print("\n" + agent.get_timing_summary())

        # Always write timing to output.txt
        agent.write_timing_to_file(args.query)

        if not args.no_execute and result.get('execution_result') is not None:
            print("\n📊 QUERY RESULTS:")
            print("="*80)
            print("Query executed successfully - results shown above (object returned in JSON)")

        print("\n📝 GENERATED SQL QUERY:")
        print("="*80)
        print(result['query'])
        print("="*80)

        # NEW: Print viz dimension hint
        viz = result.get('viz', {})
        print("\n📈 VIZ SUGGESTION:")
        print("="*80)
        print(json.dumps(viz, indent=2))

        if args.debug:
            print("\n🔍 DEBUG INFO:")
            print(f"Original Query: {args.query}")
            print(f"Tables Used: {', '.join(result['required_tables'])}")
            print(f"Medical Codes Found: {result['metadata']['codes_extracted']}")
            print(f"Query Executed: {result['metadata']['executed']}")

            if args.pretty:
                print(f"\nMedical Codes Details:")
                print(json.dumps(result['medical_codes'], indent=2))

                print(f"\nTiming Details:")
                print(json.dumps(result['timing'], indent=2))

                print(f"\nFull GPT Prompt:")
                print(result['final_prompt'])
            else:
                print(f"Medical Codes: {result['medical_codes']}")

    else:
        print(f"\n❌ Pipeline Failed: {result['error']}")

        # Show timing even on failure
        if result.get('timing') and (args.timing or args.debug):
            timing = result['timing']
            print(f"\n⏱️ Timing (before error at {timing.get('error_at_stage', 'unknown')} stage):")
            print(f"Total Time: {timing.get('total_time_s', 0)}s")
            if timing.get('stages'):
                for stage, time_ms in timing['stages'].items():
                    print(f"  • {stage.replace('_', ' ').title()}: {time_ms}ms")

        if args.debug:
            if result.get('medical_codes'):
                print(f"\nMedical codes extracted: {json.dumps(result['medical_codes'], indent=2)}")
            if result.get('required_tables'):
                print(f"Tables identified: {', '.join(result['required_tables'])}")
            if result.get('viz'):
                print(f"Viz: {json.dumps(result['viz'], indent=2)}")


if __name__ == "__main__":
    main()

# Example usage:
# python main_agent.py --query "Find top 10 providers prescribing alecensa" --debug --timing
# python main_agent.py --query "Count patients with diabetes in 2024" --pretty 
# python main_agent.py --query "Show stroke medication trends by month" --debug
# python main_agent.py --query "Count patients with hypertension" --no-execute  # Generate only
# python main_agent.py --query "Top 5 diabetes medications by volume" --timing  # Show performance
