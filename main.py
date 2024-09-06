from langchain_experimental.agents import create_csv_agent
import pandas as pd
from langchain_openai import OpenAI
import streamlit as st

def main():
    
    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    
    OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key:", type="password")
    
    if csv_file and OPENAI_API_KEY:
        df = pd.read_csv(csv_file)
        st.write(df)

        csv_file.seek(0) # resetting the file buffer
        
        agent = create_csv_agent(
            OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
            csv_file,
            verbose=True,
            allow_dangerous_code=True
        )

        user_question = st.text_input("Ask a question about your CSV:")
        
        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.invoke(user_question)["output"])

if __name__ == "__main__":
    main()