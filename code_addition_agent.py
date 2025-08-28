import os
import json
import openai
import argparse
import sys
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

load_dotenv(override=True)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found in environment variables")
    sys.exit(1)

openai.api_key = OPENAI_API_KEY

class CodeAdditionAgent:
    """
    Agent that processes medical codes, queries, and user input to generate
    a refined final query suitable for NLP-to-SQL processing.
    """
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    def extract_items_from_codes(self, medical_codes: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract drugs and medical codes from the medical codes dictionary.
        """
        items = {"drugs": [], "codes": []}
        
        # Check for drug_names field (legacy support)
        if medical_codes.get("drug_names"):
            for drug in medical_codes["drug_names"]:
                if drug.get("brand"):
                    items["drugs"].append(drug["brand"])
                elif drug.get("generic"):
                    items["drugs"].append(drug["generic"])
        
        # Extract medical codes
        code_types = ["icd10", "icd9", "cpt", "hcpcs"]
        for code_type in code_types:
            if medical_codes.get(code_type):
                for code_item in medical_codes[code_type]:
                    if isinstance(code_item, dict) and code_item.get("code"):
                        items["codes"].append(f"{code_type.upper()}: {code_item['code']}")
                    elif isinstance(code_item, str):
                        items["codes"].append(f"{code_type.upper()}: {code_item}")
        
        return items
    
    def extract_additional_items(self, user_input: str) -> Dict[str, List[str]]:
        """
        Extract additional codes, drugs, or medical terms from user input using OpenAI.
        """
        prompt = f"""
        Extract any medical items (codes, drug names, medical terms, hospital names, etc.) mentioned in this user input:
        "{user_input}"
        
        Return a JSON object with:
        {{
            "drugs": ["list of drug names found"],
            "medical_codes": {{
                "icd10": ["list of ICD-10 codes found"],
                "icd9": ["list of ICD-9 codes found"],
                "cpt": ["list of CPT codes found"],
                "hcpcs": ["list of HCPCS codes found"]
            }},
            "medical_terms": ["list of medical conditions/terms found"],
            "other_items": ["list of other relevant items like hospital names, locations, etc."],
            "intent": "brief description of what the user wants to add/modify"
        }}
        
        If nothing is found in a category, return empty list.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Extract medical items accurately from user input. Focus on actionable items that can be added to a database query."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Warning: Error extracting items from user input: {e}")
            return {"drugs": [], "medical_codes": {"icd10": [], "icd9": [], "cpt": [], "hcpcs": []}, "medical_terms": [], "other_items": [], "intent": ""}
    
    def generate_refined_query(self, 
                             original_query: str, 
                             refined_query: str, 
                             medical_codes: Dict[str, Any], 
                             user_input: str) -> Dict[str, str]:
        """
        Generate the final refined query incorporating user additions.
        """
        # Extract existing drugs and codes from medical codes
        existing_items = self.extract_items_from_codes(medical_codes)
        
        # Extract additional items from user input
        extracted_items = self.extract_additional_items(user_input)
        
        prompt = f"""
        Create a simple, clear database query by incorporating user additions into the existing refined query.
        Also provide a brief explanation of what data the SQL query will display.

        Current Information:
        - Refined Query: "{refined_query}"
        - Existing Drugs: {existing_items["drugs"]}
        - Existing Medical Codes: {existing_items["codes"]}
        - User Input: "{user_input}"
        - Extracted Items: {json.dumps(extracted_items, indent=2)}

        Requirements:
        1. Keep the query simple and direct for NLP-to-SQL processing
        2. Use plain, straightforward language
        3. Include all relevant items (drugs, medical codes, conditions) explicitly in the query (NO DUPLICATES)
        4. Avoid complex medical terminology
        5. Structure should be clear: "[action] [items] by [metric]" or similar
        6. Remove any duplicate drug names, medical codes, or items
        7. Incorporate both drugs and medical codes when adding new items from user input

        Examples of good simple queries:
        - "Top hospitals for diabetes patients by volume"
        - "Most prescribed drugs aspirin, ibuprofen by count"
        - "Best performing clinics for heart surgery by success rate"
        - "Top 10 hospitals in California for cancer treatment by patient volume"
        - "GLP-1 prescriptions Ozempic, Victoza, Trulicity, Byetta with HCPCS J3490 by volume"
        - "Diabetes patients with ICD-10 E11 and medications metformin, insulin by provider"

        IMPORTANT: Return ONLY a valid JSON object with exactly this structure:
        {{
            "final_query": "the refined query incorporating user additions with no duplicates",
            "data_explanation": "1-2 line explanation of what data the SQL will display"
        }}

        Example data explanations:
        - "Comparison of prescription counts for various medications"
        - "Hospital rankings showing patient volumes for specific medical conditions"
        - "Usage statistics comparing different items by frequency or volume"
        """
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Create simple, direct database queries suitable for NLP-to-SQL processing. Keep language clear and straightforward. Always return valid JSON with both final_query and data_explanation fields. Remove any duplicate items."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            response_content = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(response_content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract from potential markdown
                if "```json" in response_content:
                    json_start = response_content.find("```json") + 7
                    json_end = response_content.find("```", json_start)
                    response_content = response_content[json_start:json_end].strip()
                    result = json.loads(response_content)
                else:
                    raise
            
            final_query = result.get("final_query", "").strip()
            data_explanation = result.get("data_explanation", "Query results showing requested medical data").strip()
            
            # Remove quotes if AI wrapped the query
            if final_query.startswith('"') and final_query.endswith('"'):
                final_query = final_query[1:-1]
                
            return {
                "final_query": final_query,
                "data_explanation": data_explanation
            }
            
        except Exception as e:
            print(f"Error generating refined query: {e}")
            # Fallback: simple concatenation
            return {
                "final_query": f"{refined_query} {user_input}",
                "data_explanation": "Query results showing requested medical data"
            }
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to process the complete request and return the final query.
        """
        try:
            original_query = request_data.get("original_query", "")
            refined_query = request_data.get("refined_query", "")
            medical_codes = request_data.get("medical_codes", {})
            user_input = request_data.get("user_input", "")
            
            if not refined_query:
                return {
                    "status": "error",
                    "error": "refined_query is required",
                    "final_query": original_query
                }
            
            if not user_input:
                # If no user input, still incorporate existing medical codes into the query
                existing_items = self.extract_items_from_codes(medical_codes)
                
                # Create a final query that includes existing medical codes and drugs
                if existing_items["drugs"] or existing_items["codes"]:
                    all_items = existing_items["drugs"] + existing_items["codes"]
                    if all_items:
                        final_query = f"{refined_query} {', '.join(all_items)}"
                    else:
                        final_query = refined_query
                else:
                    final_query = refined_query
                
                return {
                    "status": "success",
                    "final_query": final_query,
                    "data_explanation": "Query results based on refined search criteria with existing medical codes",
                    "original_query": original_query,
                    "refined_query": refined_query,
                    "user_input": "",
                    "format": request_data.get("format", "text")
                }
            
            query_result = self.generate_refined_query(
                original_query=original_query,
                refined_query=refined_query,
                medical_codes=medical_codes,
                user_input=user_input
            )
            
            return {
                "status": "success",
                "final_query": query_result["final_query"],
                "data_explanation": query_result["data_explanation"],
                "original_query": original_query,
                "refined_query": refined_query,
                "user_input": user_input,
                "format": request_data.get("format", "text")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "final_query": request_data.get("refined_query", request_data.get("original_query", ""))
            }

def main():
    """CLI interface for the Code Addition Agent"""
    parser = argparse.ArgumentParser(description='Code Addition Agent - Process medical queries with user additions')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                      help='Input JSON string or file path containing the request data')
    parser.add_argument('--user-input', '-u', type=str,
                      help='Additional user input text (overrides JSON user_input field)')
    parser.add_argument('--pretty', '-p', action='store_true',
                      help='Pretty print the output JSON')
    
    args = parser.parse_args()
    
    try:
        # Parse input
        if args.input.startswith('{'):
            # Direct JSON string
            request_data = json.loads(args.input)
        else:
            # File path
            with open(args.input, 'r') as f:
                request_data = json.load(f)
        
        # Override user_input if provided via command line
        if args.user_input:
            request_data["user_input"] = args.user_input
        
        # Process request
        agent = CodeAdditionAgent()
        result = agent.process_request(request_data)
        
        # Output result
        if args.pretty:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result))
            
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()