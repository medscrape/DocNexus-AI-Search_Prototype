#!/usr/bin/env python3
"""
Chart Classification CLI Tool
Usage: python chart_classifier_cli.py "your query here"
"""

import os
import json
import sys
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv(override=True)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")

# Chart categories
CHART_CATEGORIES = {
    "ONE_DIMENSIONAL_CHARTS": ['pie-chart', 'donut-chart', 'funnel-chart'],
    "TWO_DIMENSIONAL_TIME_SERIES_CHARTS": ['line-chart', 'area-chart'],
    "TWO_DIMENSIONAL_NON_TIME_SERIES_CHARTS": ['vertical-bar-chart', 'horizontal-bar-chart', 'column-chart'],
    "THREE_DIMENSIONAL_TIME_SERIES_CHARTS": ['multi-series-line-chart'],
    "THREE_DIMENSIONAL_NON_TIME_SERIES_CHARTS": ['grouped-bar-chart', 'stacked-bar-chart', 'stacked-column-chart', 'bubble-chart', 'heat-map-geospatial', 'bubble-map'],
    "FOUR_DIMENSIONAL_TIME_SERIES_CHARTS": ['animated-scatter-plot'],
    "MULTI_DIMENSIONAL_CHARTS": ['sankey-diagram', 'stacked-area-categories']
}

class ChartClassificationCLI:
    def __init__(self):
        if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key. Set OPENAI_API_KEY in .env file")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = MODEL_NAME

    def create_system_prompt(self):
        return f"""You are a chart classification expert. Analyze user queries and suggest the most appropriate chart type(s) AND the column names needed for the data.

Available chart types by category:

ONE_DIMENSIONAL_CHARTS (single metric/count):
- pie-chart, donut-chart, funnel-chart

TWO_DIMENSIONAL_TIME_SERIES_CHARTS (trends over time):
- line-chart, area-chart

TWO_DIMENSIONAL_NON_TIME_SERIES_CHARTS (category comparisons):
- vertical-bar-chart, horizontal-bar-chart, column-chart

THREE_DIMENSIONAL_TIME_SERIES_CHARTS (multiple series over time):
- multi-series-line-chart

THREE_DIMENSIONAL_NON_TIME_SERIES_CHARTS (complex comparisons):
- grouped-bar-chart, stacked-bar-chart, stacked-column-chart, bubble-chart, heat-map-geospatial, bubble-map

FOUR_DIMENSIONAL_TIME_SERIES_CHARTS (complex time analysis):
- animated-scatter-plot

MULTI_DIMENSIONAL_CHARTS (flows/relationships):
- sankey-diagram, stacked-area-categories

For suggested_columns, think about what data columns would be needed:
- x_axis: Main category/dimension for horizontal axis
- y_axis: Values/metrics for vertical axis  
- categories: Grouping/segmentation column
- values: Numerical values to display

Examples:
- "Sales by region" → categories: "region", values: "sales_amount"
- "Revenue over time" → x_axis: "date", y_axis: "revenue"
- "Patient count by condition" → categories: "condition", values: "patient_count"

Return ONLY valid JSON:
{{
    "recommended_charts": ["chart-type-1", "chart-type-2"],
    "primary_category": "CATEGORY_NAME",
    "reasoning": "Brief explanation",
    "suggested_columns": {{
        "x_axis": "column_name_for_x_axis",
        "y_axis": "column_name_for_y_axis", 
        "categories": "column_name_for_categories",
        "values": "column_name_for_values"
    }}
}}"""

    def classify_query(self, query):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": f"Classify this query: {query}"}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response from AI"}
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description='Classify NLP queries into chart types')
    parser.add_argument('query', nargs='?', help='Query to classify')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--test', '-t', action='store_true', help='Run test with sample queries')
    
    args = parser.parse_args()
    
    try:
        classifier = ChartClassificationCLI()
        
        if args.test:
            # Test mode with sample queries
            test_queries = [
                "Tell me how many patients are there with NMOSD and also on Enspryng",
                "Show sales trends over last 6 months", 
                "Compare revenue by product category and region",
                "Display market share distribution",
                "Patient demographics by age group and treatment type"
            ]
            
            print("🧪 Testing Chart Classifier")
            print("=" * 50)
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n{i}. Query: {query}")
                result = classifier.classify_query(query)
                print(f"   Result: {json.dumps(result, indent=6)}")
                
        elif args.interactive:
            # Interactive mode
            print("🎯 Interactive Chart Classifier (type 'quit' to exit)")
            print("=" * 50)
            
            while True:
                query = input("\nEnter query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                    
                if query:
                    result = classifier.classify_query(query)
                    print(json.dumps(result, indent=2))
                    
        elif args.query:
            # Single query mode
            print(f"🎯 Classifying: {args.query}")
            print("=" * 50)
            result = classifier.classify_query(args.query)
            print(json.dumps(result, indent=2))
            
        else:
            # Default: classify the NMOSD query
            default_query = "Tell me how many patients are there with NMOSD and also on Enspryng"
            print(f"🎯 Default Test Query: {default_query}")
            print("=" * 50)
            result = classifier.classify_query(default_query)
            print(json.dumps(result, indent=2))
            
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()