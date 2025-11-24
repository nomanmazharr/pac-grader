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
from llm_setup import llm, llm_grader
from logging_config import logger


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

def save_json(data, filename, folder="test_assignments_and_mappings"):
    """Save Python dict or list to a JSON file."""
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f" Saved JSON to {file_path}")

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
        logger.error(f"Error extracting text from page {page_num}: {e}")
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
    You are a professional examiner who grades student answers **strictly and objectively** according to the provided model data.
    You must rely entirely on the *explicit* information from the model answers and marking criteria.
    Do not assume, infer, or fabricate any information not present in the model data.

    ### Input Data
    - Questions: \n {questions} \n
    - Model answers and criteria: {model_data}
    - Mappings: \n {mappings} \n 
    - Student answers: \n {chunks} \n

    ### Fundamental Rules
    - You must **not** grade based on wording similarity or keyword matches.
    - You must **only** use sentence-level or paragraph-level meaning comparisons.
    - If the meaning or idea in the student’s text does not fully align with the corresponding model point, award **zero** marks for that point.
    - If the annotation explicitly states partial marks (e.g., “half for each item covered”), you may apply that proportion exactly as written.
    - If the annotation does *not* mention partial credit, do **not** assign it.
    - If uncertain whether an idea matches, always choose **zero** — never assume correctness.

    ### Grading Process
    1. For each question:
       - Use "maximum_marks" as the absolute total; never exceed it.
       - Use "marking_criteria" as the sole authority for awarding marks.
       - Follow marking criteria to have an understanding of how many marks each point or para carry.

    2. Analyze each annotation point one-by-one (internal verification only)
       - Internally locate the model answer sentence(s) that define this annotation point. **Do not** copy or output model answer text anywhere in the final JSON or comment.
       - Convert that model point into a short internal checklist of the distinct conceptual elements required (2-6 items max).
       - From the student answer, identify up to **3 consecutive** lines (earliest occurrence) that might satisfy this point. If none exist → score 0 for this point.
       - For each checklist item, mark internally:
           - YES: the student text explicitly provides this element (quote the exact student phrase as internal evidence), or
           - NO: the element is missing or incorrect.
       - **Do not** combine non-consecutive fragments or unrelated sentences to satisfy a single checklist item. Patchwork matches are invalid.

    3. Scoring rules (final decision)
       - If **any** checklist item = NO → score = 0 for this annotation point (unless the annotation explicitly allows partial credit).
       - If **all** checklist items = YES:
           - If annotation specifies fractional/partial marks (e.g., "half for each"), apply that fraction exactly.
           - If annotation does **not** allow partial credit, award the full marks assigned to that annotation point.
       - Never invent partial splits beyond those explicitly stated in marking criteria.
       - If uncertain at any stage → default to 0 (no guessing).

    4. Evidence selection policy
       - `correct_lines`: include the exact student lines (up to 3) that directly supported the awarded point. Keep punctuation/formatting identical.
       - `correct_words`: from those `correct_lines`, include 2–6 word verbatim snippets that show the idea (only if present).
       - If multiple student lines could match the same annotation, choose the earliest matching consecutive block.
       - Do not include model-answer text in `correct_lines` or `comment`.

    5. No-leak / no-inference clause
       - Do not infer unstated facts, complete missing logic, or supply missing premises.
       - Ambiguous or only-topically-related answers receive 0 unless the annotation explicitly allows partial credit.
       - Do not defend or justify inferred marks — only produce the JSON with `comment`, `correct_lines`, and `correct_words` as required.

    6. Final verification (internal)
       - Confirm every awarded mark maps to a single annotation point.
       - Confirm no annotation point counted twice.
       - Confirm totals ≤ maximum_marks.
       - Confirm output will contain no model answer text.
       - Always answer in 0.5 or full like 1 never in between.

    ### Feedback
    - `comment`: 2–3 concise lines summarizing:
        - Which specific ideas or points were missing.
        - One sentence of actionable advice.
    - Avoid fluff, compliments, or model answer copying.

    ### Special Conditions
    - If the student's answer is blank, unmapped, or unrelated → score 0.
    - If any data is missing from student text for that question → treat as unanswered.
    - Never exceed the “maximum_marks” under any circumstance.
    - Never use reasoning or details not explicitly in model_data.

    ### Output Format
    Return **only** a valid JSON object, with no extra commentary, Markdown, or explanations:

    {{
      "grades": [
        {{
          "question_number": same as input,
          "score": <float or integer>,
          "total_marks": <integer>,
          "comment": "string",
          "correct_lines": ["string", "string"],
          "correct_words": ["string", "string"]
        }}
      ]
    }}
    """
)


grade_chain = grade_prompt | llm_grader

def grade_student(student_pdf_path, student_name, questions_path, model_answers_path, question_number, student_pages):
    """Grade a student's PDF and save results to CSV."""
    try:
        # student_pdf_path = os.path.join(input_dir, f"{student_name}.pdf")
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
        save_json(student_chunks, f'{student_name}_data.json')
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
        save_json(parsed_output, f'{student_name}_mappings.json')
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