import streamlit as st
import psycopg2
import pandas as pd
import csv
from io import StringIO
import json
import datetime
import openai
from openai import OpenAIError, BadRequestError, ConflictError, InternalServerError, NotFoundError

# Set up the OpenAI API client
OPENAI_API_KEY='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'
MODEL_NAME = "gpt-4-1106-preview"  # You can choose from the available OpenAI models
openai.api_key = OPENAI_API_KEY  # Ensure you have set this environment variable or set the key directly

class Database:
    def __init__(self, host, database, user, password):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.conn.cursor()
            print("Connected to PostgreSQL database!")
        except (psycopg2.Error, Exception) as error:
            print(f"Error connecting to PostgreSQL database: {error}")

    def get_tables(self):
        try:
            self.cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            tables = [row[0] for row in self.cursor.fetchall()]
            return tables
        except (psycopg2.Error, Exception) as error:
            print(f"Error retrieving tables: {error}")
            return []

    def get_schema(self, table_name):
        try:
            self.cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'")
            schema = self.cursor.fetchall()
            schema_str = f"Table: {table_name}\n"
            schema_str += "\n".join([f"{col[0]} ({col[1]})" for col in schema])
            return schema_str
        except (psycopg2.Error, Exception) as error:
            print(f"Error retrieving schema: {error}")
            return None

    def execute_query(self, query):
        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return results
        except (psycopg2.Error, Exception) as error:
            print(f"Error executing query: {error}")
            return None
        
##### dict function starts here
    def generate_data_dictionary(self, table_name):
        try:
            # Fetch sample data from the table
            self.cursor.execute(f"SELECT * FROM {table_name} LIMIT 1000")
            sample_data = self.cursor.fetchall()
            
            # Get column names and data types from the cursor description
            columns = [(desc[0], desc[1]) for desc in self.cursor.description]
            
            # Create a dictionary to store column information
            data_dict = {}
            for i, column in enumerate(columns):
                column_name = column[0]
                data_type = column[1]
                
                # Analyze sample data to infer additional information
                sample_values = [row[i] for row in sample_data]
                unique_values = list(set(sample_values))
                
                # Convert date objects to string representations
                unique_values = [str(value) if isinstance(value, datetime.date) else value for value in unique_values]
                
                # Calculate min and max values only if there are non-None values
                non_none_values = [value for value in sample_values if value is not None]
                min_value = min(non_none_values) if non_none_values else None
                max_value = max(non_none_values) if non_none_values else None
                
                # Convert date objects to string representations for min and max values
                min_value = str(min_value) if isinstance(min_value, datetime.date) else min_value
                max_value = str(max_value) if isinstance(max_value, datetime.date) else max_value
                
                # Store column information in the dictionary
                data_dict[column_name] = {
                    "data_type": data_type,
                    "unique_values": unique_values,
                    "min_value": min_value,
                    "max_value": max_value
                }
            
            # Check if the data dictionary table exists, create it if necessary
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_dictionaries (
                    table_name VARCHAR(255) PRIMARY KEY,
                    data_dict JSONB
                )
            """)
            
            # Insert or update the data dictionary in the database
            self.cursor.execute("""
                INSERT INTO data_dictionaries (table_name, data_dict)
                VALUES (%s, %s)
                ON CONFLICT (table_name) DO UPDATE SET data_dict = EXCLUDED.data_dict
            """, (table_name, json.dumps(data_dict, default=str)))
            
            self.conn.commit()
            
            return data_dict
        
        except (psycopg2.Error, Exception) as error:
            print(f"Error generating data dictionary: {error}")
            return None

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
# end create datadictionary and samples    
# end class database

# Function to create a prompt for OpenAI based on the schema, data dictionary, and user question
def create_openai_prompt(schema, data_dict, question):
    prompt = f"""Translate this question into a SQL query based on the following table schema and data dictionary, output the sql query with out any comment:

Schema:
{schema}

Data Dictionary:
{json.dumps(data_dict, indent=2)}

Question:
{question}
"""
    return prompt

#### ask openai start
def ask_openai(prompt, model="text-davinci-003", temperature=0.0, max_tokens=150):

    """
    Sends a prompt to the OpenAI model and retrieves the response.
    Args:
        prompt (str): The prompt to send to the model.
        temperature (float): Controls randomness in the generation. Lower values mean less random completions.
        max_tokens (int): The maximum number of tokens to generate in the completion.
    
    Returns:
        str: The response from the OpenAI model or an error message.
    """
    try:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop="\n"
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    return None
#### ask openai end   

# start sql_query

def execute_sql_query(sql_query, db):
    try:
        # Fetch data from database
        df = db.execute_query(sql_query)
        
        if not df.empty:
            num_rows = len(df)
            
            if num_rows <= 5:
                # Convert dataframe to a single paragraph description
                explanation = convert_dataframe_to_paragraph(df)
                st.write(explanation)
            elif num_rows <= 24:
                # Display results in Streamlit as a table
                st.table(df)
            else:
                # Display only the first 24 rows and offer a download as CSV for the rest
                st.warning("The query returned more than 24 rows. Displaying the first 24 rows.")
                st.table(df.head(24))
                
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name="query_results.csv",
                    mime="text/csv",
                )
                
                # Request a comment from OpenAI
                comment_prompt = f"""Here are the results of a SQL query:
{df.head(10).to_string(index=False)}  # Reduce the data printed to make the prompt smaller.
Can you provide a concise explanation of these results in Portuguese? Limit your suggestions for further analysis to two key points.
"""
                comment_response = openai.Completion.create(
                    model=MODEL_NAME,
                    prompt=comment_prompt,
                    max_tokens=150
                )
                comment = comment_response.choices[0].text.strip()

                st.write("Comments:")
                st.write(comment)
        else:
            st.warning("The query did not return any results. Please check your question and try again.")
    
    except Exception as e:
        st.error(f"An error occurred while processing your request: {e}")

# Helper function that needs to be implemented
def convert_dataframe_to_paragraph(df):
    # Analyze numerical columns
    numerical_analysis = ""
    for col in df.select_dtypes(include=[np.number]):
        col_mean = df[col].mean()
        col_median = df[col].median()
        col_std = df[col].std()  
        numerical_analysis += f"The average of {col} is {col_mean:.2f}, median is {col_median:.2f}, and standard deviation is {col_std:.2f}. "
  
    # Analyze categorical columns
    categorical_analysis = ""
    for col in df.select_dtypes(include=['object', 'category']):
        top_value = df[col].mode()[0]
        categorical_analysis += f"The most common value for {col} is {top_value}. "
    
    # Combine analyses into a single paragraph
    paragraph = f"Based on the query results, {numerical_analysis}{categorical_analysis}"
    
    # Suggest further methodologies
    suggestions = "For numerical data, consider regression analysis or time-series forecasting. For categorical data, look into frequency analysis or trend detection."
    
    return paragraph, suggestions
# end sql_qeury
        
# end ask 
def main():
    st.title("FASTSQL - A quick Report tool")
    # PostgreSQL connection details
    host = "localhost"
    database = "ANS"
    user = "postgres"
    password = "Jtx1970"
    db = None
    # Assuming `Database` class and other necessary functions are already defined in the script...
    db = Database(host, database, user, password)
    db.connect()

    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    try:
        tables = db.get_tables() if db.conn else []
        selected_table = st.sidebar.selectbox("Select a table", tables)
        schema_str = db.get_schema(selected_table) if selected_table else ""
        data_dict = db.generate_data_dictionary(selected_table) if selected_table else {}

        if schema_str and data_dict:
            st.sidebar.header("Database Schema")
            st.sidebar.text(schema_str)  # Do not print the entire dictionary, just reference it's available

            user_input = st.text_area("Enter your question in natural language:")
            if st.button("Send"):
                # Generate SQL query prompt
                sql_prompt = create_openai_prompt(schema_str, data_dict, user_input)
                
                # Choose a model suited to code-related prompts for generating SQL
                sql_query = ask_openai(sql_prompt, model="davinci-codex")  

                if sql_query:
                    st.code(sql_query)
                    results_df = db.execute_query(sql_query)
                    
                    # Check if the result set is not empty and act accordingly
                    if not results_df.empty:
                        if len(results_df) <= 5:
                            st.table(results_df)
                            paragraph, suggestions = convert_dataframe_to_paragraph(results_df)
                            explanation_prompt = f"Explain these results:\n{paragraph}\n{suggestions}"
                            explanation = ask_openai(explanation_prompt, model="text-davinci-003")
                            st.write(explanation)
                        else:
                            # For larger result sets, provide a download link for a CSV file
                            csv_buffer = StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="Download query results as CSV",
                                data=csv_buffer.getvalue(),
                                file_name="query_results.csv",
                                mime="text/csv",
                            )
                            # Display a summary table of the first 24 rows
                            st.write("Displaying the first 24 rows of the query results:")
                            st.table(results_df.head(24))
                    else:
                        st.warning("The SQL query did not return any results. Please check your question and try again.")

                else:
                    st.error("Failed to generate SQL query. Please try again.")
                                                    
            elif db.conn:
                st.warning("Please select a table to start querying.")
        else:
            st.error("Failed to connect to the database.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        if db and db.conn:
            db.close()

if __name__ == "__main__":
    main()