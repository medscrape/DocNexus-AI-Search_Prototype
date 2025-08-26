#!/usr/bin/env python3
"""
Flask API Server for NLP to ClickHouse
======================================
Provides REST API endpoints for generating and executing ClickHouse queries from natural language.

Endpoints:
- POST /GetSQLQuery: Generate SQL query from natural language
- POST /ExecuteQuery: Execute SQL query on ClickHouse database
"""

import json
import time
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables (override existing ones)
load_dotenv(override=True)

# Import our existing agents
from v1_agent import NLPToClickHouseAgent
from clickhouse_agent import ClickHouseAgent
from clarifying_agent import QueryClarifyingAgent
from column_simple import get_medical_codes

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize agents globally
nlp_agent = None
clickhouse_agent = None
clarifying_agent = None

def initialize_agents():
    """Initialize agents on first request"""
    global nlp_agent, clickhouse_agent, clarifying_agent
    if nlp_agent is None:
        nlp_agent = NLPToClickHouseAgent()
    if clickhouse_agent is None:
        clickhouse_agent = ClickHouseAgent()
    if clarifying_agent is None:
        clarifying_agent = QueryClarifyingAgent()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'NLP to ClickHouse API',
        'timestamp': time.time()
    })

@app.route('/get-clarifying-questions', methods=['POST'])
def get_clarifying_questions():
    """
    Stage 1A: Get clarifying questions for a natural language query
    
    Request Body:
    {
        "query": "Find top hospitals for diabetes patients"
    }
    
    Response:
    {
        "success": true,
        "original_query": "Find top hospitals for diabetes patients",
        "questions": "To better understand your request: 1. Do you want rankings by patient volume...?",
        "needs_clarification": true
    }
    """
    try:
        # Initialize agents
        initialize_agents()
        
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: query'
            }), 400
        
        user_query = data['query'].strip()
        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        # Get clarifying questions from the agent
        questions = clarifying_agent._get_questions(user_query)
        
        # Check if clarification is needed
        needs_clarification = "QUERY_CLEAR" not in questions
        
        response = {
            'success': True,
            'original_query': user_query,
            'questions': questions if needs_clarification else "Query is clear and ready for processing.",
            'needs_clarification': needs_clarification
        }
        
        # Log the request
        app.logger.info(f"Clarifying Questions Generated - Query: {user_query[:100]}... - Needs Clarification: {needs_clarification}")
        
        return jsonify(response), 200
    
    except Exception as e:
        app.logger.error(f"Error in get_clarifying_questions: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/process-clarification', methods=['POST'])
def process_clarification():
    """
    Stage 1B: Process user's clarification responses and create refined query
    
    Request Body:
    {
        "original_query": "Find top hospitals for diabetes patients",
        "clarification": "I want rankings by patient volume for 2024 in California"
    }
    
    Response:
    {
        "success": true,
        "original_query": "Find top hospitals for diabetes patients",
        "clarification_provided": "I want rankings by patient volume for 2024 in California",
        "refined_query": "Find top 10 hospitals in California by diabetes patient volume for 2024",
        "medical_codes": {
            "icd10": [{"code": "E11", "description": "Type 2 diabetes mellitus"}],
            "icd9": [{"code": "25000", "description": "Diabetes mellitus without mention of complication"}],
            "cpt": [],
            "hcpcs": [],
            "loinc": [],
            "snomed": [],
            "jcodes": []
        }
    }
    """
    try:
        # Initialize agents
        initialize_agents()
        
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        if not data or 'original_query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: original_query'
            }), 400
        
        original_query = data['original_query'].strip()
        clarification = data.get('clarification', '').strip()
        
        if not original_query:
            return jsonify({
                'success': False,
                'error': 'Original query cannot be empty'
            }), 400
        
        # Process the clarification
        if clarification and clarification.lower() != 'proceed':
            # Combine original query with clarification
            combined_prompt = f"""Combine these into one clear query:
Original: "{original_query}"
Clarification: "{clarification}"

Return refined query:"""

            try:
                response = requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers={'Authorization': f'Bearer {clarifying_agent.api_key}', 'Content-Type': 'application/json'},
                    json={
                        'model': 'gpt-4o-mini',
                        'messages': [{'role': 'user', 'content': combined_prompt}],
                        'temperature': 0.1,
                        'max_tokens': 100
                    }
                )
                if response.status_code == 200:
                    refined = response.json()['choices'][0]['message']['content'].strip()
                    refined_query = refined.strip('"').strip("'")
                else:
                    refined_query = f"{original_query}. {clarification}"
            except:
                refined_query = f"{original_query}. {clarification}"
        else:
            refined_query = original_query
        
        # Apply expansion to the refined query
        final_query = clarifying_agent._expand_query(refined_query)
        
        # Get medical codes for the refined query
        medical_codes = get_medical_codes(final_query)
        
        response = {
            'success': True,
            'original_query': original_query,
            'clarification_provided': clarification,
            'refined_query': final_query,
            'medical_codes': medical_codes
        }
        
        # Log the request
        app.logger.info(f"Clarification Processed - Original: {original_query[:50]}... - Final: {final_query[:50]}...")
        
        return jsonify(response), 200
    
    except Exception as e:
        app.logger.error(f"Error in process_clarification: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/GetSQLQuery', methods=['POST'])
def get_sql_query():
    """
    Generate ClickHouse SQL query from natural language
    
    Request Body:
    {
        "query": "Count patients with diabetes",
        "execute": false  // optional, default false
    }
    
    Response:
    {
        "success": true,
        "sql_query": "SELECT count(DISTINCT patient_id) FROM medical_blue WHERE...",
        "medical_codes": {...},
        "required_tables": [...],
        "timing": {...},
        "metadata": {...}
    }
    """
    try:
        # Initialize agents
        initialize_agents()
        
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: query'
            }), 400
        
        user_query = data['query'].strip()
        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        # Generate SQL query (without execution)
        result = nlp_agent.generate_and_execute_query(user_query, execute=False)
        
        if result['success']:
            response = {
                'success': True,
                'sql_query': result['query'],
                'medical_codes': result['medical_codes'],
                'required_tables': result['required_tables'],
                'timing': result['timing'],
                'metadata': {
                    'original_query': result['metadata']['original_query'],
                    'tables_used': result['metadata']['tables_used'],
                    'codes_extracted': result['metadata']['codes_extracted']
                }
            }
            
            # Log the request for monitoring
            app.logger.info(f"SQL Query Generated - Query: {user_query[:100]}... - Time: {result['timing']['total_time_ms']}ms")
            
            return jsonify(response), 200
        else:
            app.logger.error(f"SQL Generation Failed - Query: {user_query} - Error: {result['error']}")
            return jsonify({
                'success': False,
                'error': result['error'],
                'timing': result.get('timing', {}),
                'medical_codes': result.get('medical_codes', {}),
                'required_tables': result.get('required_tables', [])
            }), 500
    
    except Exception as e:
        app.logger.error(f"Unexpected error in GetSQLQuery: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/ExecuteQuery', methods=['POST'])
def execute_query():
    """
    Execute SQL query on ClickHouse database
    
    Request Body:
    {
        "sql_query": "SELECT count(*) FROM medical_blue",
        "format": "json"  // optional, default is raw
    }
    
    Response:
    {
        "success": true,
        "results": [...],
        "execution_time_ms": 150,
        "row_count": 1000,
        "columns": [...],
        "metadata": {...}
    }
    """
    try:
        # Initialize agents
        initialize_agents()
        
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        if not data or 'sql_query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: sql_query'
            }), 400
        
        sql_query = data['sql_query'].strip()
        if not sql_query:
            return jsonify({
                'success': False,
                'error': 'SQL query cannot be empty'
            }), 400
        
        output_format = data.get('format', 'raw')
        
        # Execute the query
        start_time = time.time()
        result = clickhouse_agent.execute(sql_query)
        execution_time = time.time() - start_time
        
        if result is not None:
            # Extract result data
            rows = result.result_rows
            columns = result.column_names
            
            # Format results based on requested format
            if output_format.lower() == 'json':
                # Convert to list of dictionaries
                formatted_results = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        row_dict[col] = row[i] if i < len(row) else None
                    formatted_results.append(row_dict)
            else:
                # Raw format - list of lists
                formatted_results = [list(row) for row in rows]
            
            response = {
                'success': True,
                'results': formatted_results,
                'execution_time_ms': round(execution_time * 1000, 2),
                'row_count': len(rows),
                'columns': columns,
                'metadata': {
                    'query': sql_query,
                    'format': output_format,
                    'executed_at': time.time()
                }
            }
            
            # Log successful execution
            app.logger.info(f"Query Executed Successfully - Rows: {len(rows)} - Time: {execution_time*1000:.2f}ms")
            
            return jsonify(response), 200
        else:
            app.logger.error(f"Query Execution Failed - Query: {sql_query}")
            return jsonify({
                'success': False,
                'error': 'Query execution failed - check query syntax and permissions',
                'execution_time_ms': round(execution_time * 1000, 2),
                'metadata': {
                    'query': sql_query,
                    'executed_at': time.time()
                }
            }), 400
    
    except Exception as e:
        app.logger.error(f"Unexpected error in ExecuteQuery: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/GetAndExecuteQuery', methods=['POST'])
def get_and_execute_query():
    """
    Combined endpoint: Generate SQL from natural language AND execute it
    
    Request Body:
    {
        "query": "Count patients with diabetes",
        "format": "json"  // optional
    }
    
    Response:
    {
        "success": true,
        "original_query": "Count patients with diabetes",
        "sql_query": "SELECT count(DISTINCT patient_id) FROM medical_blue WHERE...",
        "results": [...],
        "generation_time_ms": 5000,
        "execution_time_ms": 150,
        "total_time_ms": 5150,
        "row_count": 1,
        "columns": [...],
        "medical_codes": {...},
        "required_tables": [...],
        "viz": {
            "dimension": 1,
            "chart": "pie",
            "rationale": "Single categorical GROUP BY with aggregates"
        },
        "metadata": {...}
    }
    """
    try:
        # Initialize agents
        initialize_agents()
        
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: query'
            }), 400
        
        user_query = data['query'].strip()
        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        output_format = data.get('format', 'raw')
        
        # Generate and execute SQL query
        result = nlp_agent.generate_and_execute_query(user_query, execute=True)
        
        if result['success']:
            # Extract execution results
            execution_result = result.get('execution_result')
            
            if execution_result is not None:
                # Format execution results
                rows = execution_result.result_rows
                columns = execution_result.column_names
                
                if output_format.lower() == 'json':
                    # Convert to list of dictionaries
                    formatted_results = []
                    for row in rows:
                        row_dict = {}
                        for i, col in enumerate(columns):
                            row_dict[col] = row[i] if i < len(row) else None
                        formatted_results.append(row_dict)
                else:
                    # Raw format - list of lists
                    formatted_results = [list(row) for row in rows]
                
                response = {
                    'success': True,
                    'original_query': user_query,
                    'sql_query': result['query'],
                    'results': formatted_results,
                    'generation_time_ms': result['timing']['stages'].get('parallel_analysis', 0) + result['timing']['stages'].get('gpt_generation', 0),
                    'execution_time_ms': result['timing']['stages'].get('query_execution', 0),
                    'total_time_ms': result['timing']['total_time_ms'],
                    'row_count': len(rows),
                    'columns': columns,
                    'medical_codes': result['medical_codes'],
                    'required_tables': result['required_tables'],
                    'viz': result.get('viz', {}),
                    'metadata': {
                        'tables_used': result['metadata']['tables_used'],
                        'codes_extracted': result['metadata']['codes_extracted'],
                        'executed_at': time.time(),
                        'format': output_format
                    }
                }
            else:
                # Query generated but execution failed
                response = {
                    'success': False,
                    'error': 'SQL query generated successfully but execution failed',
                    'original_query': user_query,
                    'sql_query': result['query'],
                    'results': [],
                    'generation_time_ms': result['timing']['stages'].get('parallel_analysis', 0) + result['timing']['stages'].get('gpt_generation', 0),
                    'execution_time_ms': result['timing']['stages'].get('query_execution', 0),
                    'total_time_ms': result['timing']['total_time_ms'],
                    'medical_codes': result['medical_codes'],
                    'required_tables': result['required_tables'],
                    'viz': result.get('viz', {})
                }
                return jsonify(response), 400
            
            # Log successful completion
            app.logger.info(f"Full Pipeline Completed - Query: {user_query[:100]}... - Rows: {len(rows)} - Total Time: {result['timing']['total_time_ms']}ms")
            
            return jsonify(response), 200
        else:
            app.logger.error(f"Pipeline Failed - Query: {user_query} - Error: {result['error']}")
            return jsonify({
                'success': False,
                'error': result['error'],
                'original_query': user_query,
                'timing': result.get('timing', {}),
                'medical_codes': result.get('medical_codes', {}),
                'required_tables': result.get('required_tables', [])
            }), 500
    
    except Exception as e:
        app.logger.error(f"Unexpected error in GetAndExecuteQuery: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/tables', methods=['GET'])
def list_tables():
    """
    List available tables and their information
    
    Response:
    {
        "tables": {
            "medical_blue": {
                "purpose": "...",
                "columns": [...],
                "key_columns": {...}
            },
            ...
        }
    }
    """
    try:
        initialize_agents()
        
        # Get table information from the agent
        table_info = nlp_agent.table_agent.get_all_table_info()
        
        # Add schema information
        tables_with_schema = {}
        for table_name, info in table_info.items():
            tables_with_schema[table_name] = {
                **info,
                'schema': nlp_agent.table_schemas.get(table_name, {})
            }
        
        return jsonify({
            'tables': tables_with_schema
        }), 200
    
    except Exception as e:
        app.logger.error(f"Error in list_tables: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /health',
            'POST /get-clarifying-questions',
            'POST /process-clarification',
            'POST /GetSQLQuery',
            'POST /ExecuteQuery', 
            'POST /GetAndExecuteQuery',
            'GET /tables'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Flask API Server...")
    print("📊 Available endpoints:")
    print("   GET  /health                      - Health check")
    print("   POST /get-clarifying-questions    - Get clarifying questions for NLP query")
    print("   POST /process-clarification       - Process clarification and refine query")
    print("   POST /GetSQLQuery                 - Generate SQL from natural language")
    print("   POST /ExecuteQuery                - Execute SQL query on ClickHouse")
    print("   POST /GetAndExecuteQuery          - Combined: Generate + Execute")
    print("   GET  /tables                      - List available tables")
    print()
    print("🔗 Server will be available at: http://localhost:5001")
    print("📖 Use Content-Type: application/json for POST requests")
    
    # Run the Flask app
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),  # Allow external connections
        port=int(os.getenv('FLASK_PORT', 5001)),   # Using port from env or default 5001
        debug=os.getenv('FLASK_DEBUG', 'True').lower() == 'true',  # Enable debug mode for development
        threaded=True    # Handle multiple requests concurrently
    )