import pandas as pd
import sqlite3
from datetime import datetime
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# --- 0. Configure API Key ---
# IMPORTANT: Set your API key as an environment variable before running.
# For OpenAI: OPENAI_API_KEY
try:
    # Using OpenAI GPT
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print("OpenAI API configured successfully.")
except Exception as e:
    print(f"Error: API key not configured. Please set the OPENAI_API_KEY environment variable. {e}")
    exit()


# --- 1. Database Setup ---

def create_unified_database():
    """
    Loads data from combined CSV,
    and loads everything into an in-memory SQLite database.
    """
    try:
        # Load the combined data file
        main_df = pd.read_csv('results/combined.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure 'combined.csv' is in the 'results' directory.")
        return None, None

    # For date queries, ensure 'created_at' is in a queryable format
    main_df['created_at'] = pd.to_datetime(main_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create an in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    # Load the DataFrame into the database
    main_df.to_sql('disputes', conn, index=False, if_exists='replace')

    print("In-memory SQLite database created and populated successfully.")
    return conn, main_df.columns.tolist()

# --- 2. LLM Interaction for SQL Generation ---

def get_sql_from_llm(user_query, schema):
    """
    Sends the user query and DB schema to the LLM to get a SQL query.
    """
    # Get today's date to inject into the prompt for time-sensitive queries
    today_date = datetime.now().strftime('%Y-%m-%d')

    prompt = f"""
    You are an expert SQL query writer. Based on the database schema and a user question,
    generate a single, executable SQLite query.

    Database Schema:
    Table Name: disputes
    Columns: {', '.join(schema)}

    Instructions:
    - The `predicted_category` column contains values 'DUPLICATE_CHARGE', 'FRAUD', 'FAILED_TRANSACTION', 'REFUND_PENDING' and 'OTHERS'.
    - The `status` column contains values 'resolved' or 'unresolved'.
    - The `suggested_action` column contains values 'Auto-refund', 'Manual review', 'Escalate to bank', 'Mark as potential fraud', 'Ask for more info'
    - For any questions involving "today", use the date '{today_date}'.
    - The `created_at` column is in 'YYYY-MM-DD HH:MM:SS' format.
    - Only output the SQL query, with no explanation or other text.

    User Question: "{user_query}"

    SQL Query:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can also use "gpt-4" for better results
            messages=[
                {"role": "system", "content": "You are an expert SQL query writer. Only respond with the SQL query, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        content = response.choices[0].message.content
        print("generated content:",content)
        sql_query = content.strip().replace("```sql", "").replace("```", "")
        return sql_query
    except Exception as e:
        return f"Error generating SQL: {e}"

# --- 3. LLM Interaction for Final Answer ---

def get_answer_from_llm(user_query, query_result_df):
    """
    Sends the result of the SQL query to the LLM to get a natural language answer.
    """
    prompt = f"""
    You are an AI assistant for a dispute resolution team.
    You have been asked a question and have received the answer in a structured format.
    Formulate a friendly, concise, natural-language response to the user.

    Original Question: "{user_query}"

    Data Result:
    {query_result_df.to_string(index=False)}

    Your Friendly Answer:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can also use "gpt-4" for better results
            messages=[
                {"role": "system", "content": "You are a friendly AI assistant for a dispute resolution team. Provide concise, helpful answers based on the data provided."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating final answer: {e}"

# --- 4. Main CLI Loop ---
def main():

    conn, schema = create_unified_database()

    if conn and schema:
        print("\n--- AI Dispute Assistant ---")
        print("Ask questions about your dispute data. Type 'exit' to quit.")

        while True:
            user_input = input("\nYour question: ")
            if user_input.lower() == 'exit':
                break

            # Step 1: Generate SQL from user query
            print("-> Generating SQL query...")
            sql_query = get_sql_from_llm(user_input, schema)
            print(f"   Generated SQL: {sql_query}")

            # Step 2: Execute SQL and get results
            try:
                result_df = pd.read_sql(sql_query, conn)
                print("-> Executed SQL successfully.")

                # Step 3: Generate final answer
                if result_df.empty:
                    final_answer = "I couldn't find any data matching your query."
                else:
                    print("-> Generating final response...")
                    final_answer = get_answer_from_llm(user_input, result_df)

                print(f"\nAssistant: {final_answer}")

            except Exception as e:
                print(f"An error occurred: {e}")

        conn.close()
        print("\nSession ended. Goodbye!")
    else:
        print("Failed to initialize the database. Please check your CSV files.")