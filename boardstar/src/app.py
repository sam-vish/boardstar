import os
import sys
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dotenv import load_dotenv
import streamlit as st
from src.rag import RAG
from src.agents import TestGenerator, TestEvaluator
from src.database import get_classes, get_subjects, get_chapters

# Load environment variables from .env file
load_dotenv()

# Set the Google API key from the environment variable
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

def main():
    st.title("AI Test Generator and Evaluator (Gemini Pro)")

    # Initialize RAG system
    database_path = os.path.join(project_root, "database")
    rag = RAG(database_path)
    with st.spinner("Initializing RAG system..."):
        rag.create_vector_store()

    if not rag.vector_store:
        st.error("Failed to initialize RAG system. Please check your database directory and documents.")
        logger.error("RAG system initialization failed")
        return

    # Initialize agents
    test_generator = TestGenerator()
    test_evaluator = TestEvaluator()

    # Sidebar for input selection
    st.sidebar.header("Select Test Parameters")
    selected_class = st.sidebar.selectbox("Class", get_classes())
    selected_subject = st.sidebar.selectbox("Subject", get_subjects(selected_class))
    selected_chapter = st.sidebar.selectbox("Chapter", get_chapters(selected_class, selected_subject))
    num_questions = st.sidebar.slider("Number of Questions", 1, 10, 5)

    # Generate test
    if st.sidebar.button("Generate Test"):
        query_results = rag.query(f"{selected_class} {selected_subject} {selected_chapter}", k=1)
        if not query_results:
            st.error("No relevant content found. Please check your selection.")
            return
        context = query_results[0][0]
        questions = test_generator.generate_test(context, num_questions)
        st.session_state.questions = questions
        st.session_state.show_questions = True

    # Display questions and collect answers
    if st.session_state.get('show_questions', False):
        st.header("Answer the Following Questions")
        
        with st.form(key='quiz_form'):
            answers = []
            for i, question in enumerate(st.session_state.questions.split('\n\n')):
                st.markdown(f"**Question {i+1}:**")
                lines = question.split('\n')
                st.write(lines[0])  # Question text
                options = [line for line in lines[1:5]]  # A, B, C, D options with labels
                answer = st.radio(f"Select your answer for Question {i+1}", options, key=f"answer_{i}")
                answers.append(answer)

            submit_button = st.form_submit_button(label='Submit Test')

        if submit_button:
            evaluation = test_evaluator.evaluate_test(st.session_state.questions, "\n".join(answers))
            st.write(evaluation)
            st.session_state.show_questions = False  # Hide questions after submission

if __name__ == "__main__":
    main()