import json
import os
import requests
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# Load environment variables (override existing ones)
load_dotenv(override=True)

class MedicalTableIdentificationAgent:
    """
    Schema-driven Table Identification Agent using GPT-4o mini
    No hardcoded examples - purely based on table schemas and semantics
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Get API key from environment variables or parameter
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Define tables based on actual schema structure from CSVs
        self.table_catalog = {
            'medical_blue': {
                'purpose': 'Medical claims and clinical encounters',
                'core_data': 'Patient medical history, procedures, and diagnoses',
                'schema_categories': {
                    'temporal': 'Year, month, service dates for temporal analysis',
                    'demographics': 'Patient age, gender, and geographic data',
                    'clinical_codes': 'ICD-10 diagnosis codes, CPT/HCPCS procedure codes, DRG codes',
                    'providers': 'Provider NPIs, referring physicians, billing entities',
                    'facility': 'Place of service, hospital/clinic information',
                    'financial': 'Claim amounts, payment information'
                },
                'analytical_capabilities': [
                    'Count patients with specific medical conditions',
                    'Track procedures performed over time',
                    'Analyze disease prevalence and demographics',
                    'Study clinical outcomes and patterns',
                    'Measure procedure volumes by provider or facility'
                ],
                'key_identifiers': 'Diagnosis codes (ICD), Procedure codes (CPT/HCPCS), Patient demographics'
            },
            
            'pharmacy_blue': {
                'purpose': 'Prescription drug claims and pharmacy data',
                'core_data': 'Medication dispensing, prescriptions, and drug utilization',
                'schema_categories': {
                    'temporal': 'Year, month, prescription fill dates',
                    'demographics': 'Patient age, gender for drug utilization',
                    'drug_details': 'Generic name, brand name, NDC codes, strength, dosage',
                    'dispensing': 'Days supply, quantity, refills, mail order flag',
                    'providers': 'Prescriber NPI, pharmacy NPI, prescriber specialty',
                    'financial': 'Drug costs, patient pay, insurance coverage'
                },
                'analytical_capabilities': [
                    'Track medication utilization and adherence',
                    'Analyze drug prescribing patterns',
                    'Monitor pharmacy network performance',
                    'Study drug costs and payer coverage',
                    'Measure growth trends for specific medications'
                ],
                'key_identifiers': 'Drug names, NDC codes, Prescription patterns, Pharmacy data'
            },
            
            'as_providers_v1': {
                'purpose': 'Healthcare provider registry and organizational directory',
                'core_data': 'Individual providers, medical organizations, and referral networks',
                'schema_categories': {
                    'identifiers': 'Type 1 NPI (individual), Type 2 NPIs (organizational), Tax IDs',
                    'provider_info': 'Name, credentials, gender, specialties, taxonomy codes',
                    'organization': 'Organization names, hospital affiliations, medical groups',
                    'location': 'Practice addresses, cities, states, contact information',
                    'network': 'Referral patterns, provider relationships, affiliations'
                },
                'analytical_capabilities': [
                    'Identify top providers for specific services',
                    'Analyze referral networks and patterns',
                    'Find providers by specialty or location',
                    'Track organizational performance',
                    'Map provider-facility relationships'
                ],
                'key_identifiers': 'Provider NPIs, Organization names, Specialties, Referral data'
            },
            
            'as_providers_referrals_v2': {
                'purpose': 'Provider referral patterns and relationships with clinical context',
                'core_data': 'Referral networks between providers with associated diagnoses, procedures, and financial data',
                'schema_categories': {
                    'referral_network': 'Primary and referring provider relationships (Type 1 and Type 2 NPIs)',
                    'provider_details': 'Provider names, specialties, hospital affiliations',
                    'geographic': 'Provider locations with coordinates, cities, states, postal codes',
                    'clinical_context': 'Diagnosis codes and descriptions, procedure codes and descriptions',
                    'temporal': 'Date-based referral tracking and trends',
                    'financial': 'Total claim charges and line item charges',
                    'volume_metrics': 'Patient counts for referral relationships'
                },
                'analytical_capabilities': [
                    'Analyze referral patterns between specific provider types',
                    'Track referral volumes for specific diagnoses or procedures',
                    'Map geographic referral networks and care coordination',
                    'Study financial impact of referral relationships',
                    'Identify top referring and receiving providers by specialty',
                    'Analyze referral trends over time by clinical conditions',
                    'Map hospital-to-hospital referral networks',
                    'Study specialty-specific referral patterns'
                ],
                'key_identifiers': 'Primary/Referring NPIs, Provider specialties, Diagnosis/Procedure codes, Hospital affiliations'
            }
        }
        
        # Updated semantic patterns including referral-specific terms
        self.semantic_patterns = {
            'needs_diagnosis': ['diagnosis', 'condition', 'disease', 'icd', 'patients with'],
            'needs_procedure': ['procedure', 'surgery', 'surgical', 'surgeon','cpt', 'hcpcs', 'performed', 'operation', 'bariatric'],
            'needs_drug': ['drug', 'medication', 'prescription', 'prescribed', 'pharmacy'],
            'needs_provider': ['provider', 'doctor', 'physician', 'npi', 'organization', 'hospital'],
            'needs_referral': ['referral', 'referred', 'referring', 'sent', 'referral pattern', 'care coordination'],
            'needs_temporal': ['trend', 'growth', 'over time', 'between', 'year'],
            'needs_demographic': ['age', 'gender', 'male', 'female', 'demographic'],
            'needs_financial': ['cost', 'payment', 'payer', 'insurance', 'coverage', 'charge'],
            'needs_geographic': ['location', 'geographic', 'city', 'state', 'coordinates', 'map'],
            'needs_network_analysis': ['network', 'relationship', 'connection', 'pattern', 'flow']
        }
    
    def build_gpt_prompt(self, query: str) -> str:
        """Build a comprehensive prompt for GPT-4o mini"""
        
        prompt = """You are a medical data expert helping identify which database tables are needed to answer queries.

AVAILABLE TABLES:

"""
        # Add table descriptions
        for table_name, info in self.table_catalog.items():
            prompt += f"TABLE: {table_name}\n"
            prompt += f"Purpose: {info['purpose']}\n"
            prompt += f"Contains: {info['core_data']}\n"
            prompt += "Schema includes:\n"
            for category, description in info['schema_categories'].items():
                prompt += f"  - {category}: {description}\n"
            prompt += "Can be used for:\n"
            for capability in info['analytical_capabilities']:
                prompt += f"  - {capability}\n"
            prompt += f"Key identifiers: {info['key_identifiers']}\n\n"
        
        prompt += """
COMBINATION GUIDELINES:
- Use medical_blue + pharmacy_blue when query involves both diagnoses AND medications
- Use medical_blue + as_providers_v1 when query asks about providers performing procedures
- Use pharmacy_blue + as_providers_v1 when query asks about organizations/providers prescribing drugs
- Use pharmacy_blue + medical_blue when query involves payer/insurance analysis for drugs
- Use as_providers_referrals_v2 when query asks about referral patterns, provider relationships, or care coordination
- Use as_providers_referrals_v2 + medical_blue when analyzing referrals for specific clinical conditions
- Use as_providers_referrals_v2 + as_providers_v1 when needing detailed provider information with referral patterns
- Use as_providers_referrals_v2 alone for geographic referral analysis or specialty-specific referral patterns

TASK: Identify which table(s) are needed for this query.
Consider what data elements are required to answer the question.

QUERY: """ + query + """

Return ONLY a JSON array of table names needed, like: ["table1"] or ["table1", "table2"]
Do not include any explanation, just the JSON array.
"""
        return prompt
    
    def identify_tables_with_gpt(self, query: str) -> List[str]:
        """Use GPT-4o mini to identify tables"""
        if not self.api_key:
            return self.identify_tables_semantic(query)
        
        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'gpt-4o-mini',
                    'messages': [
                        {
                            'role': 'system', 
                            'content': 'You are a database expert. Return only JSON arrays, no explanations.'
                        },
                        {
                            'role': 'user',
                            'content': self.build_gpt_prompt(query)
                        }
                    ],
                    'temperature': 0,
                    'max_tokens': 50
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                # Parse the JSON response
                tables = json.loads(content)
                return tables if isinstance(tables, list) else ['medical_blue']
            else:
                print(f"API Error: {response.status_code}")
                return self.identify_tables_semantic(query)
                
        except Exception as e:
            print(f"Error calling GPT: {e}")
            return self.identify_tables_semantic(query)
    
    def identify_tables_semantic(self, query: str) -> List[str]:
        """Semantic analysis fallback when GPT is not available"""
        query_lower = query.lower()
        tables_needed = set()
        
        # Analyze query semantics
        needs = {}
        for need_type, patterns in self.semantic_patterns.items():
            needs[need_type] = any(pattern in query_lower for pattern in patterns)
        
        # Determine tables based on semantic needs
        if needs['needs_diagnosis'] or needs['needs_procedure']:
            tables_needed.add('medical_blue')
        
        if needs['needs_drug']:
            tables_needed.add('pharmacy_blue')
        
        if needs['needs_provider'] or needs['needs_referral'] or needs['needs_network_analysis']:
            if needs['needs_referral'] or needs['needs_network_analysis'] or needs['needs_geographic']:
                tables_needed.add('as_providers_referrals_v2')
            else:
                tables_needed.add('as_providers_v1')
        
        # Handle combinations
        if needs['needs_drug'] and needs['needs_diagnosis']:
            tables_needed.update(['medical_blue', 'pharmacy_blue'])
        
        if needs['needs_provider'] and needs['needs_procedure']:
            tables_needed.update(['medical_blue', 'as_providers_v1'])
        
        if needs['needs_drug'] and needs['needs_provider']:
            if 'organization' in query_lower or 'hospital' in query_lower:
                tables_needed.update(['pharmacy_blue', 'as_providers_v1'])
        
        if needs['needs_financial'] and needs['needs_drug']:
            tables_needed.update(['pharmacy_blue', 'medical_blue'])
            
        # Referral-specific combinations
        if needs['needs_referral'] and needs['needs_diagnosis']:
            tables_needed.update(['as_providers_referrals_v2', 'medical_blue'])
            
        if needs['needs_referral'] and needs['needs_provider']:
            tables_needed.update(['as_providers_referrals_v2', 'as_providers_v1'])
        
        # Default if nothing detected
        if not tables_needed:
            tables_needed.add('medical_blue')
        
        return list(tables_needed)
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific table"""
        return self.table_catalog.get(table_name, {})

    def get_all_table_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all tables"""
        return self.table_catalog

    def validate_tables(self, tables: List[str]) -> List[str]:
        """Validate that table names exist in catalog"""
        valid_tables = []
        for table in tables:
            if table in self.table_catalog:
                valid_tables.append(table)
            else:
                print(f"Warning: Table '{table}' not found in catalog")
        return valid_tables if valid_tables else ['medical_blue']  # Default fallback

    def get_table_schema_summary(self, table_name: str) -> str:
        """Get a concise schema summary for a table"""
        if table_name not in self.table_catalog:
            return f"Table '{table_name}' not found"
        
        info = self.table_catalog[table_name]
        summary = f"Table: {table_name}\n"
        summary += f"Purpose: {info['purpose']}\n"
        summary += f"Key identifiers: {info['key_identifiers']}\n"
        summary += "Capabilities: " + ", ".join(info['analytical_capabilities'][:3]) + "...\n"
        return summary

    def validate_configuration(self) -> bool:
        """Validate that the agent is properly configured"""
        if not self.api_key:
            print("Warning: No API key provided, using semantic analysis only")
            return False
        if not self.table_catalog:
            print("Error: No table catalog loaded")
            return False
        return True
    
    def identify_tables(self, query: str, use_gpt: bool = True) -> List[str]:
        """Main interface to identify tables with validation"""
        if use_gpt and self.api_key:
            tables = self.identify_tables_with_gpt(query)
        else:
            tables = self.identify_tables_semantic(query)
        
        # Validate the results
        return self.validate_tables(tables)
    
    def explain_tables(self) -> str:
        """Get a detailed explanation of all tables"""
        explanation = "="*80 + "\n"
        explanation += "MEDICAL DATA WAREHOUSE - TABLE CATALOG\n"
        explanation += "="*80 + "\n\n"
        
        for table_name, info in self.table_catalog.items():
            explanation += f"📊 TABLE: {table_name.upper()}\n"
            explanation += "-"*40 + "\n"
            explanation += f"Purpose: {info['purpose']}\n"
            explanation += f"Core Data: {info['core_data']}\n\n"
            
            explanation += "Schema Categories:\n"
            for cat, desc in info['schema_categories'].items():
                explanation += f"  • {cat}: {desc}\n"
            
            explanation += "\nAnalytical Capabilities:\n"
            for cap in info['analytical_capabilities']:
                explanation += f"  ✓ {cap}\n"
            
            explanation += f"\nKey Identifiers: {info['key_identifiers']}\n"
            explanation += "\n" + "="*80 + "\n\n"
        
        return explanation

    def generate_sample_queries(self) -> Dict[str, List[str]]:
        """Generate sample queries to test the as_providers_referrals_v2 table"""
        return {
            'as_providers_referrals_v2': [
                # Basic referral analysis
                "Show me the top 10 referring providers by patient volume",
                "Which hospitals receive the most referrals?",
                "What are the most common diagnoses in referrals?",
                
                # Geographic analysis
                "Map referral patterns between different cities",
                "Show referral flows from rural to urban areas",
                "Which states have the highest inter-state referral rates?",
                
                # Specialty analysis  
                "Which specialties refer most to cardiology?",
                "Show referral patterns between primary care and specialists",
                "What procedures drive the most referrals?",
                
                # Financial analysis
                "What is the average claim charge for referred patients?",
                "Which referral relationships generate the highest revenue?",
                "Show cost analysis of referral patterns by diagnosis",
                
                # Temporal analysis
                "How have referral patterns changed over time?",
                "Show seasonal trends in referrals by specialty",
                "Track referral volume growth between specific provider pairs",
                
                # Network analysis
                "Identify the most connected providers in the referral network",
                "Show care coordination patterns for specific conditions",
                "Map referral networks for cancer patients"
            ]
        }

if __name__ == "__main__":
    # Initialize agent with hardcoded API key
    agent = MedicalTableIdentificationAgent()
    
    # Display table catalog
    print(agent.explain_tables())
    
    # Test queries including new referral queries
    test_queries = [
        "Tell me the top hospitals for alecensa",
        "Count patients with diabetes",  
        "Show medication trends over time",
        "Show me referral patterns between cardiologists",
        "Which providers refer most patients to Mayo Clinic?",
        "Map geographic referral networks for cancer patients",
        "What are the financial impacts of referral relationships?"
    ]
    
    print("\nQUERY ANALYSIS RESULTS\n" + "="*80)
    
    for i, query in enumerate(test_queries, 1):
        tables = agent.identify_tables(query, use_gpt=False)  # Using semantic for demo
        print(f"\nQuery {i}: {query}")
        print(f"Tables Required: {', '.join(tables)}")
        for table in tables:
            schema_info = agent.get_table_schema_summary(table)
            print(f"  - {schema_info}")
            
    # Show sample queries for the new table
    print("\n\nSAMPLE QUERIES FOR as_providers_referrals_v2\n" + "="*80)
    sample_queries = agent.generate_sample_queries()
    
    for table, queries in sample_queries.items():
        print(f"\n📊 {table.upper()} - Sample Queries:")
        print("-"*50)
        for i, query in enumerate(queries, 1):
            print(f"{i:2d}. {query}")
else:
    # When imported, don't run the test automatically
    pass