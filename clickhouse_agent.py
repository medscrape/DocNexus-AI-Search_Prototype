import time
import re
import os
from clickhouse_connect import get_client
from dotenv import load_dotenv

# Load environment variables (override existing ones)
load_dotenv(override=True)


class ClickHouseAgent:
    def __init__(self, host=None, username=None, password=None, port=None, secure=None):
        # Get ClickHouse configuration from environment variables
        self.host = host or os.getenv("CLICKHOUSE_HOST")
        self.username = username or os.getenv("CLICKHOUSE_USERNAME", "default")
        self.password = password or os.getenv("CLICKHOUSE_PASSWORD")
        self.port = int(port or os.getenv("CLICKHOUSE_PORT", 8443))
        self.secure = secure if secure is not None else os.getenv("CLICKHOUSE_SECURE", "True").lower() == "true"
        
        # Validate required credentials
        if not self.host:
            raise ValueError("CLICKHOUSE_HOST environment variable is required")
        if not self.password:
            raise ValueError("CLICKHOUSE_PASSWORD environment variable is required")
        
        self.client = get_client(
            host=self.host, 
            username=self.username, 
            password=self.password,
            port=self.port, 
            secure=self.secure
        )
    
    def is_read_only(self, query):
        """Check if query is read-only"""
        query_upper = query.strip().upper()
        
        # Reject forbidden operations
        forbidden = ['DROP', 'CREATE', 'ALTER', 'DELETE', 'UPDATE', 'INSERT', 
                    'TRUNCATE', 'REPLACE', 'GRANT', 'REVOKE', 'SET', 'KILL']
        
        for word in forbidden:
            if word in query_upper:
                return False
        
        # Allow only SELECT, DESCRIBE, SHOW, EXPLAIN
        allowed_starts = ['SELECT', 'DESCRIBE', 'DESC', 'SHOW', 'EXPLAIN', 'WITH']
        
        return any(query_upper.startswith(start) for start in allowed_starts)
    
    def execute(self, query):
        """Execute query and return results with timing"""
        print(f"Query: {query}")
        
        # Check if read-only
        if not self.is_read_only(query):
            print("REJECTED: Only read-only queries allowed")
            return None
        
        # Execute with timing
        start_time = time.time()
        try:
            result = self.client.query(query)
            execution_time = time.time() - start_time
            
            # Print results
            print(f"Execution time: {execution_time:.3f}s")
            print(f"Rows returned: {len(result.result_rows)}")
            print(f"Columns: {result.column_names}")
            print("-" * 50)
            
            # Print data (limit to 20 rows)
            for i, row in enumerate(result.result_rows[:20]):
                print(f"Row {i+1}: {row}")
            
            if len(result.result_rows) > 20:
                print(f"... and {len(result.result_rows) - 20} more rows")
            
            print("=" * 50)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Error after {execution_time:.3f}s: {e}")
            return None



if __name__ == "__main__":

    agent = ClickHouseAgent()
    test_queries = [
        "DESCRIBE TABLE medical_blue",
        "SELECT COUNT(*) FROM medical_blue LIMIT 5",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        agent.execute(query)
        input("Press Enter to continue...")  # Pause between queries