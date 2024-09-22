from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
from pydantic import Field, BaseModel
import re
import os
import random

# Ensure you have set the GOOGLE_API_KEY environment variable
os.environ['GOOGLE_API_KEY'] = "AIzaSyDmf0d09V7jGsuN-kfZ6Di-bF0LbCyH7_I"  # Replace with your actual API key

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool] = Field(default_factory=list)
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\nThought: I now know the result of the action.\n"
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        if llm_output.strip().startswith("```python"):
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )
        if "GenerateQuestion" in llm_output:
            return AgentAction(tool="GenerateQuestion", tool_input=llm_output, log=llm_output)
        match = re.search(r"Action: (.*?)[\n]*Action Input:[\s]*(.*)", llm_output, re.DOTALL)
        if match:
            action = match.group(1).strip()
            action_input = match.group(2)
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        raise ValueError(f"Could not parse LLM output: `{llm_output}`")

class TestGenerator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    def _generate_single_question(self, context):
        prompt = f"""Generate a single multiple-choice question based on this context:

{context}

The question should have 4 options (A, B, C, D) with one correct answer. Format the question exactly as follows:

Q: [Question text]
A: [Option A]
B: [Option B]
C: [Option C]
D: [Option D]
Correct Answer: [A/B/C/D]

Do not include any additional text, code, or explanations. Only provide the formatted question."""

        response = self.llm.predict(prompt)
        return response.strip()

    def generate_test(self, context, num_questions=5):
        all_questions = []
        for _ in range(num_questions):
            question = self._generate_single_question(context)
            all_questions.append(question)
        return "\n\n".join(all_questions)

    def format_questions(self, questions_text):
        # This method is no longer needed as we're ensuring proper formatting in _generate_single_question
        return questions_text

class TestEvaluator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

    def _evaluate_single_answer(self, question, student_answer, correct_answer):
        prompt = f"""Evaluate this answer to the given question:

Question: {question}
Student's Answer: {student_answer}
Correct Answer: {correct_answer}

Provide a brief evaluation and score (0-100):"""
        
        return self.llm.predict(prompt)

    def evaluate_test(self, questions, answers):
        # Parse questions and answers
        parsed_questions = re.findall(r'Q: (.*?)(?:\nA:|$)', questions, re.DOTALL)
        parsed_answers = answers.split('\n')
        correct_answers = re.findall(r'Correct Answer: ([A-D])', questions)

        if len(parsed_questions) != len(parsed_answers) or len(parsed_questions) != len(correct_answers):
            return "Error: Mismatch in the number of questions and answers."

        evaluations = []
        total_score = 0

        for q, sa, ca in zip(parsed_questions, parsed_answers, correct_answers):
            evaluation = self._evaluate_single_answer(q, sa, ca)
            evaluations.append(evaluation)
            
            # Extract score from evaluation
            score_match = re.search(r'(\d+)/100', evaluation)
            if score_match:
                total_score += int(score_match.group(1))

        average_score = total_score / len(parsed_questions)
        
        result = "Test Evaluation:\n\n"
        for i, (q, sa, ca, eval) in enumerate(zip(parsed_questions, parsed_answers, correct_answers, evaluations), 1):
            result += f"Question {i}:\n{q}\nYour Answer: {sa}\nCorrect Answer: {ca}\nEvaluation: {eval}\n\n"
        
        result += f"Overall Score: {average_score:.2f}/100"
        
        return result