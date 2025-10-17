import fitz  
import os
import json
from pathlib import Path
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from logging_config import logger
from llm_setup import llm
from datetime import datetime


class SubAnswer(BaseModel):
    question_number: str = Field(description="Subquestion number like '1.1' or 'a)'")
    answer: Optional[str] = Field(description="The model answer content if available for that section; omit if the content belongs to its sub-sections")
    marking_criteria: Optional[str] = Field(None, description="Marking criteria details if available; omit if it belongs to sub sections")
    total_marks_available: Optional[str] = Field(None, description="Total marks available for this part; omit if belong to subsections")
    maximum_marks: Optional[str] = Field(None, description="Maximum full marks (if stated); omit if belong to sub sections")
    sub_answers: Optional[List["SubAnswer"]] = Field(
        None,
        description="Nested sub-answers if the subquestion has further divisions like a), b), etc."
    )

class ModelAnswerExtraction(BaseModel):
    question_title: str = Field(description="Main question title, e.g., 'Question 1'")
    description: Optional[str] = Field(None, description="Introductory paragraph or assumption if given")
    answers: List[SubAnswer] = Field(..., description="List of main answers or sub-sections like 1.1, 1.2, etc.")
    total_marks: Optional[str] = Field(None, description="Total marks for this main question if mentioned")

prompt_template = """
You are an expert in extracting and structuring model answers from exam marking guides.

Focus on question {question_num} and its model answers.

Given the following model answer text from PDF page(s):

{answer_text}

Instructions:
- Extract **only** Question {question_num} and its associated model answers.
- Create separate entries only if explicit subsections are present (e.g., 1.1, 1.2, a), b), A), B)).
- If subsections exist (e.g., 1.1, 1.2 or a), b)), extract each subsection's:
  - model answer
  - marking criteria
- If no subsections exist, treat the entire content as a **single entry** with question_number equal to the main question number (e.g., "1").
- **Never leave total_marks_available or maximum_marks empty.**
- **maximum_marks** and **total marks available** should only appear **once per sub-section** (even if that sub-section contains nested sub-parts).
  - **maximum_marks** and **total marks available** are usually found at the end of the marking criteria and often phrased as "Maximum", "Maximum marks", or "Maximum full marks" for maximum marks and **Total Possible Marks** or **Marks Available** for total possible marks
- Extract **marking criteria** from the marking guide section and keep it separate from the answer text.
  - The `answer` field must contain only the actual model answer content.
  - The `marking_criteria` field must contain only the marking instructions and related scoring breakdown for each instruction.
- Preserve all model answer and marking criteria content **exactly as written** - no summarizing or rewording.
- Use `sub_answers` only when a section has nested divisions like (a), (b), (i), (ii), etc.
- Always maintain the correct question hierarchy (main → sub → nested sub) based on explicit numbering or lettering.

### Output Format:
{format_instructions}
"""

# Parser for model answer extraction
parser = PydanticOutputParser(pydantic_object=ModelAnswerExtraction)

# Create the prompt with format instructions for model answers
prompt_model_answer = PromptTemplate(
    template=prompt_template,
    input_variables=["answer_text", "question_num"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create the LLM chain for model answer extraction
chain_model_answer = prompt_model_answer | llm  | parser

# Keep question extraction chain for reference (from previous code)
class SubQuestion(BaseModel):
    question_number: str = Field(description="Subquestion number like '1.1' or 'a)'")
    content: str = Field(description="Full content of the subquestion")
    marks: Optional[str] = Field(None, description="Marks for this subquestion, e.g., '5 marks'")

class QuestionExtraction(BaseModel):
    question_title: str = Field(description="Main question title (e.g., 'Question 1')")
    description: Optional[str] = Field(None, description="Introductory description or assumptions")
    questions: List[SubQuestion] = Field(..., description="List of subquestions or single main question")
    total_marks: Optional[str] = Field(None, description="Total marks for this main question")

question_parser = PydanticOutputParser(pydantic_object=QuestionExtraction)

question_prompt = PromptTemplate(
    template="""
You are an expert at extracting and structuring **exam questions** from question papers.

Your task is to process *only* question {question_num} from the given text below.

The goal is to accurately identify:
- The **main question title**
- Any **introductory description or assumptions**
- The **subquestions** (like 1.1, 1.2, a), b), etc.)
- **Marks** if mentioned (e.g., “(5 marks)” or “Total: 10 marks”)

---

### Extraction Instructions:

1. **Do not modify or paraphrase** any content.  
   → Keep the exact original wording, spacing, and formatting as in the given text.

2. If there is **introductory content or context** that:
   - **applies to all subquestions**, include it in the field `description`.
   - **applies to only specific subquestions**, include that content both in the `description` and within those specific subquestions.

3. If **no introduction or description** is given before subquestions,  
   → set `description` to `None` and include only subquestion text.

4. If the question has **no subquestions**, treat it as a single subquestion  
   → use the main question number as the subquestion id.

5. Always **preserve the original line breaks and bullet points**.

---

### Input:
{answer_text}

---

### Output Format:
{format_instructions}
""",
    input_variables=["answer_text", "question_num"],
    partial_variables={"format_instructions": question_parser.get_format_instructions()}
)

chain_question = question_prompt | llm | question_parser

def extract_text_from_pdf_pages(pdf_path: str, page_numbers: List[int]) -> str:
    """Extract text from specified PDF pages."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in sorted(page_numbers):
        if page_num > len(doc):
            raise ValueError(f"Page {page_num} does not exist in PDF")
        page = doc.load_page(page_num - 1)  # 0-indexed
        page_text = page.get_text()
        text += f"\n--- Page {page_num} ---\n{page_text}\n\n"
    
    doc.close()
    return text.strip()

def extract_single_model_answer(pdf_path: str, page_numbers: List[int], question_num: str) -> ModelAnswerExtraction:
    """
    Extract a single specific question's model answer using the chain.
    
    Args:
        pdf_path: Path to model answer PDF file
        page_numbers: List of 1-indexed page numbers
        question_num: Question number to extract (e.g., "1", "2")
    
    Returns:
        ModelAnswerExtraction: Structured model answer data
    """
    # Extract text from PDF
    pdf_text = extract_text_from_pdf_pages(pdf_path, page_numbers)
    
    if not pdf_text.strip():
        raise ValueError(f"No text found on specified pages for question {question_num}")
    
    # Use the model answer chain to extract
    result = chain_model_answer.invoke({
        "answer_text": pdf_text,
        "question_num": question_num
    })
    
    return result

def extract_single_question(pdf_path: str, page_numbers: List[int], question_num: str) -> QuestionExtraction:
    """Extract a single question from question paper (existing functionality)."""
    pdf_text = extract_text_from_pdf_pages(pdf_path, page_numbers)
    if not pdf_text.strip():
        raise ValueError(f"No text found for question {question_num}")
    
    result = chain_question.invoke({
        "answer_text": pdf_text,
        "question_num": question_num
    })
    return result

def save_extracted_data(data: object, output_path: str):
    """
    Generic save function for any Pydantic model data.
    
    Args:
        data: Pydantic model instance (QuestionExtraction, ModelAnswerExtraction, etc.)
        output_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data.model_dump(exclude_unset=True), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved data to {output_path}")

def extract_and_save_question_answer(
    question_pdf_path: str,
    question_pages: List[int],
    model_answer_pdf_path: str,
    answer_pages: List[int],
    question_num: str,
    output_dir: str = "questions_and_model_answers_json_and_scripts"
) -> tuple[str, str]:
    """
    Extract question and model answer, save to JSON files with timestamp.
    
    Returns:
        tuple: (questions_path, model_answers_path)
    """
    try:
        # Extract data
        question_data = extract_single_question(question_pdf_path, question_pages, question_num)
        model_answer_data = extract_single_model_answer(model_answer_pdf_path, answer_pages, question_num)
        
        # Generate timestamp and paths
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        question_dir = os.path.join(output_dir, "question")
        model_answer_dir = os.path.join(output_dir, "model_answer")
        Path(question_dir).mkdir(parents=True, exist_ok=True)
        Path(model_answer_dir).mkdir(parents=True, exist_ok=True)
        
        questions_path = os.path.join(question_dir, f"first_question_{timestamp}.json")
        model_answers_path = os.path.join(model_answer_dir, f"first_model_answer_{timestamp}.json")
        
        # Save using single generic function
        save_extracted_data(question_data, questions_path)
        save_extracted_data(model_answer_data, model_answers_path)
        
        print(f"✓ Extracted and saved Question {question_num}")
        return questions_path, model_answers_path
        
    except Exception as e:
        print(f"✗ Failed to extract Question {question_num}: {str(e)}")
        raise

# # Example usage
# if __name__ == "__main__":
#     try:
#         # Paths to files
        
#         model_answer_path = r"F:/PAC/Multi_Questions_scripts/6. Ltd T All_AA_E2_01-May Q4 20 marks (66 scripts) Gap Orchid DL 8 May 5pm Uk time_/ICAEW_2024_AA_Mock_Orchid_As_GAP_marked up (2024).pdf"
#         question_paper_path = r"F:/PAC/Multi_Questions_scripts/6. Ltd T All_AA_E2_01-May Q4 20 marks (66 scripts) Gap Orchid DL 8 May 5pm Uk time_/ICPR01(ME4)DEC24_Qs - Orchid_d2 viewing.pdf"
        
#         # Pages containing the content
#         question_pages = [3,4,5]  # Pages for question paper
#         answer_pages = [3,4,5]    # Pages for model answers
#         question_number = "1"

#         # question_q1 = extract_single_question(question_paper_path, question_pages, question_number)
#         # save_extracted_data(question_q1, "dummy_extraction/question1111.json")
#         # Extract Question 1 model answer
#         model_answer_q1 = extract_single_model_answer(model_answer_path, answer_pages, question_number)
#         save_extracted_data(model_answer_q1, "dummy_extraction/model_answer_2222.json")
        
#         print(f"Extracted model answer for Question 1:")
#         print(f"  Title: {model_answer_q1.question_title}")
#         print(f"  Total marks: {model_answer_q1.total_marks}")
#         for sub_answer in model_answer_q1.answers:
#             print(f"  Subpart {sub_answer.question_number}: {sub_answer.maximum_marks}")
    
#         # Extract corresponding question
        
        
#     except Exception as e:
#         print(f"Error: {str(e)}")