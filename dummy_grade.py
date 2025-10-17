from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from typing import Dict
from langchain.prompts import PromptTemplate
import json
import fitz
import re
import pandas as pd
import os
import datetime
from logging_config import logger
from llm_setup import llm, llm_grader


# Pydantic Schemas
class SubPart(BaseModel):
    question_number: str = Field(description="The identifier of the subsection or scenario (e.g., '1.1' or 'a)')")
    answer: str = Field(description="content paragraphs from the student's answer for marking criteria")

class QuestionExtraction(BaseModel):
    question: str = Field(description="The main question number (e.g., '1' or '4')")
    sub_parts: List[SubPart] = Field(description="List of subsections with their content, only if subsections like 1.1, a), A) are present")

class MappingItem(BaseModel):
    chunk_id: int = Field(..., description="Identifier of the student answer chunk.")
    mapped_question_number: str = Field(..., description="The matched question number, e.g., '1.1', or '0' if unmapped.")

class MappingList(BaseModel):
    mappings: List[MappingItem]

class GradingItem(BaseModel):
    question_number: str = Field(..., description="The number of the question/sub-question, e.g., '1.1'.")
    score: str = Field(..., description="Marks obtained by the student, e.g., '3'.")
    total_marks: str = Field(..., description="Total marks for the question, e.g., '5', from maximum_marks, only include integer value nothing else like marks and other words.")
    comment: str = Field(..., description="Feedback comment for the student, Should be concise but covering what went wrong and to the point, should not exceed three lines")
    correct_lines: List[str] = Field(..., description="Exact lines from the student's answer that are deemed correct, should be exact matching with same wording and everything")
    correct_words: List[str] = Field(..., description="Exact words from the student's answer explaining why the lines are correct.")

class GradingList(BaseModel):
    grades: List[GradingItem]


def load_json_data(questions_path, model_answers_path):
    """Load questions and model answers from JSON files."""
    try:
        with open(questions_path, 'r') as f:
            questions = json.load(f)['questions']
        with open(model_answers_path, 'r') as f:
            model_data = json.load(f)['answers']
        logger.info(f"Loaded questions from {questions_path} and model answers from {model_answers_path}")
        return questions, model_data
    except Exception as e:
        logger.error(f"Error loading JSON data: {e}")
        raise

def extract_page_text(pdf_path: str, page_num: int) -> str:
    """
    Extracts text from a specific page of the PDF using PyMuPDF.
    """
    try:
        doc = fitz.open(pdf_path)
        if page_num < 0 or page_num >= len(doc):
            return ""
        page = doc.load_page(page_num)
        text = page.get_text("text")
        doc.close()
        # Clean the text to remove headers and extra formatting
        text = re.sub(r"^\d+ /\d+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"Word Processing area.*?- use the shortcut keys to copy from the spreadsheet\s*", "", text)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from page {page_num}: {e}")
        return ""

def extract_answers(pdf_path: str, question_num: str, page_nums: List[int]) -> Dict:
    """
    Extracts and processes the answer for a given question by:
    1. Extracting text from specified pages of the student's PDF.
    2. Sending it to the LLM chain for structured answer extraction.
    
    Args:
        pdf_path (str): Path to the student's PDF file.
        question_num (str): Question number (e.g., '1', '4', etc.).
        page_nums (List[int]): List of page numbers containing the answer.

    Returns:
        Dict: Parsed model output containing question and sub_parts, 
              or an error dictionary if extraction fails.
    """
    try:
        # --- Step 1: Extract and combine text from relevant pages ---
        texts = []
        for p in page_nums:
            text = extract_page_text(pdf_path, p - 1)  # Assuming extract_page_text handles 0-indexing
            if text:
                texts.append(f"--- Page {p} ---\n{text.strip()}")
        
        answer_text = "\n\n".join(texts)

        # --- Step 2: Handle case where no content is found ---
        if not answer_text.strip():
            return {"error": f"No content found for question {question_num} on pages {page_nums}"}

        # --- Step 3: Run the LLM chain for structured extraction ---
        response = chain_answer.invoke({
            "answer_text": answer_text,
            "question_num": question_num
        })

        # --- Step 4: Return structured output ---
        student_answer = response.model_dump()
        return student_answer

    except Exception as e:
        # --- Handle unexpected errors gracefully ---
        return {"error": f"Failed to extract or parse answer for question {question_num}: {str(e)}"}


map_to_questions_parser = PydanticOutputParser(pydantic_object=MappingList)

# Prompt template for answer extraction
prompt_template = """
You are an expert in extracting and structuring student answers from exam PDFs for marking.

Focus on question {question_num} and its parts.

Given the following student answer text from a PDF page(s):

{answer_text}

Instructions:
- Identify the main question number based on the content (e.g., starts with 1.1 for question 1).
- Only create separate sub_parts if explicit subsections are present (e.g., 1.1, 1.2, a), b), A), B)).
- If subsections are present (e.g., 1.1, 1.2 or a), b)), extract each subsection's content with its id and split into paragraphs if present with proper new lines characters.
- If no subsections are present (e.g., no 1.1, 1.2, a), b), A), B)), treat the entire content as a single sub_part with id equal to the question number and include all content as given in paras or as it is.
- Focus only on the answer content, ignoring headers like 'Word Processing area'.
- Do not add or change information; extract and structure what's present.
- Alwasy remeber that only create subsections if student has specified the subsections else keep the content as a single question answer.
- Output strictly in the specified JSON format.

{format_instructions}
"""

# Parser for the output
parser = PydanticOutputParser(pydantic_object=QuestionExtraction)

# Create the prompt with format instructions
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["answer_text", "question_num"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create the LLM chain for answer extraction
chain_answer = prompt | llm | parser


map_to_questions_prompt = ChatPromptTemplate.from_template(
    """Student chunks: {chunks}\n
Questions: {questions}\n

Instructions:
- Map each student chunk to the question it most likely answers based on semantic meaning.
- Focus on the **intent** and **content** of the student chunk and question, not just exact wording.
- Each chunk must map to exactly one question number.
- If a chunk does not answer any question, assign it to '0'.
- Do NOT output explanations, schemas, or markdown. 
- Return ONLY valid JSON in the following format:

{{
  "mappings": [
    {{ "chunk_id": 1, "mapped_question_number": "1.1" }},
    {{ "chunk_id": 2, "mapped_question_number": "2.3" }},
    {{ "chunk_id": 3, "mapped_question_number": "0" }}
  ]
}}

Now produce the mappings:
"""
)


map_chain = map_to_questions_prompt | llm

grade_parser = PydanticOutputParser(pydantic_object=GradingList)


grade_prompt = ChatPromptTemplate.from_template(
    """
    You are a professional teacher who grades student answers fairly and accurately against model answers, balancing strictness with reasonable evaluation.
    Given mappings: {mappings}\nModel answers: {model_data}\nStudent answers: {chunks}\nQuestions: {questions}\n
    For each question in the model answers, compare the student's answer to the model answers with a focus on content accuracy, wording and semantic meaning.

    Award full marks only if all key points are fully and accurately covered; partial answers earn proportional marks according to the given criteria in model data based on the number of correctly and completely addressed points. Accept semantically equivalent phrasing if all details match, but penalize omissions or incomplete ideas. Always assign marks in 0.5 or full number no in between in the fractions.
    
    Always use maximum_marks from model_data as the total marks; score must be a fraction or integer out of that total, reflecting exact matches only.
    If marking criteria in model_data includes per-point breakdowns (e.g., in Unicode like ½ per item), grade each sub-point individually and sum accurately; do not round up.
    For each question, identify only the correct lines and words that precisely match the model answer's requirements and on which you have allocated the marks.
    correct_lines: Exact, unaltered lines or full paragraphs/bullets from the student's answer (use same punctuation, wording, and formatting as written by the student). Exclude any question statements, headers, or non-answer text (e.g., lines ending in '?'). Understand each line and if you see that it's a different line than the previous treat it as a different line else keep together.
    correct_words: 2-6 word phrases extracted directly from the correct_lines, in their exact original order, that capture the core reason for correctness (focus on key terms, facts, or phrases). Use full lines only if the essence can't be captured shorter; phrases must appear verbatim as in the student's text.
    Add a concise comment: Summarize what was correct (for appreciation if any), exactly what was missing or wrong (be specific to key points omitted), and brief advice. Limit to three lines maximum; cover all aspects for the student to understand the score without fluff.
    If no student text matches a question (unmapped, missing, or empty in chunks/mappings), strictly score '0', provide feedback explaining what the student should have done, referencing key elements from the model answer to guide improvement. Do not copy or borrow from model answers under any circumstances—treat as absent.

    ### Output Format
    Return **only** a single valid JSON object in the following structure (no extra text, no markdown, no explanations):

    {{
  "grades": [
    {{
      "question_number": keep the question number as it is given,
      "score": Integer or float value as score that student got,
      "total_marks": Max marks from model answers don't include key like marks and other only integer value,
      "comment": "string", 
      "correct_lines": ["string", "string"],
      "correct_words": ["string", "string"]
    }}
  ]
}}
"""
    )

grade_chain = grade_prompt | llm_grader

def grade_student(input_dir, student_name, questions_path, model_answers_path, question_number, student_pages):
    """Grade a student's PDF and save results to CSV."""
    try:
        # student_pdf_path = os.path.join(input_dir, f"{student_name}.pdf")
        student_pdf_path = input_dir
        if not os.path.exists(student_pdf_path):
            logger.error(f"Student PDF not found: {student_pdf_path}")
            return None

        # Ensure grades directory exists
        grades_dir = os.path.join("student_assignment", "grades")
        os.makedirs(grades_dir, exist_ok=True)

        # Generate output CSV path with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join(grades_dir, f"{student_name}_grades_{timestamp}.csv")

        questions, model_data = load_json_data(questions_path, model_answers_path)
   
        student_chunks = extract_answers(student_pdf_path, question_number, student_pages)
        logger.info(f"Loaded student assignment data for question number: {question_number}")
        logger.info(f"Student's Assignment: {student_chunks}")

        if not student_chunks:
            logger.error(f"No answers could be extracted for {student_name}. Skipping grading.")
            return None

        # Map to questions
        map_output = map_chain.invoke({
            "chunks": student_chunks,
            "questions": json.dumps(questions)
        })
        
        # print(map_output)
        parsed_output = json.loads(map_output.content)
        logger.info(f"Mapped question number to the student assignments: {parsed_output}")
        # Now you can access the "mappings" list
        mappings = parsed_output["mappings"]
        
        logger.info(f"Starting grading for {student_name} for question number {question_number}")
        grade_output = grade_chain.invoke({
            "mappings": mappings,
            "model_data": json.dumps(model_data),
            "chunks": student_chunks,
            "questions": json.dumps(questions)
        })
        logger.info(f"Grading done saving data into csv for: {student_name}")
        # print(grade_output)
        raw_content = grade_output.content
        cleaned_json_str = re.sub(r"^```json\n|```$", "", raw_content.strip())
        parsed_output = json.loads(cleaned_json_str)
    
        results = []


        all_questions = []
        for q in model_data:
            all_questions.append({
                "question_number": q["question_number"],
                "maximum_marks": q.get("maximum_marks", "0")
            })
        # Process graded results
        for g in parsed_output['grades']:

            question_number = g["question_number"]
            student_chunks_dict = {
                sp["question_number"]: sp for sp in student_chunks.get("sub_parts", [])
            }

            # Then you can safely do:
            chunk_text = student_chunks_dict.get(question_number)
            snippet = (
                chunk_text["answer"].split("\n")[0][:30]
                if chunk_text and chunk_text.get("answer")
                else "No answer provided"
            )
            results.append({
                "student_id": student_name,
                "question_number": g["question_number"],
                "score": g["score"],
                "total_marks": g["total_marks"],
                "comment": g["comment"],
                "correct_lines": g["correct_lines"],
                "correct_words": g["correct_words"],
                "student_answer_snippet": snippet
            })

        # Ensure all questions are covered
        graded_questions = {r["question_number"] for r in results}
        for q in all_questions:
            q_num = q["question_number"]
            if q_num not in graded_questions:
                results.append({
                    "student_id": student_name,
                    "question_number": q_num,
                    "score": "0",
                    "total_marks": q["maximum_marks"],
                    "comment": "No answer provided",
                    "correct_lines": [],
                    "correct_words": [],
                    "student_answer_snippet": "No answer provided"
                })

        # Export to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        logger.info(f"Grading complete! CSV saved to {output_csv}")
        return output_csv
    except Exception as e:
        logger.error(f"Error during grading for {student_name}: {e}")
        return None