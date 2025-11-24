
import fitz
import os
import json
from pathlib import Path
from typing import List
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from llm_setup import llm
from logging_config import logger


class SubQuestion(BaseModel):
    question_number: str = Field(description="Subquestion identifier like '1.1', 'a)', etc.")
    content: str = Field(description="Full original text of the subquestion")
    marks: str | None = Field(None, description="Marks allocation if mentioned, e.g. '5 marks'")

class QuestionExtraction(BaseModel):
    question_title: str = Field(description="Main question title, e.g. 'Question 4'")
    description: str | None = Field(None, description="Any introductory text or scenario")
    questions: List[SubQuestion] = Field(..., description="List of subquestions (or single main question)")
    total_marks: str | None = Field(None, description="Total marks for the entire question if stated")

question_parser = PydanticOutputParser(pydantic_object=QuestionExtraction)

question_prompt = PromptTemplate(
    template="""
You are an expert at extracting exam questions from PDF question papers.

Extract **only** Question {question_num} from the text below.

Rules:
- Preserve **exact original wording**, line breaks, bullet points, and formatting.
- Include any introductory scenario/description in the `description` field.
- If there are no subquestions, create one SubQuestion with the main question number.
- Capture marks exactly as written (e.g., "(6 marks)", "Total: 20 marks").

Input text:
{answer_text}

Output format:
{format_instructions}
""",
    input_variables=["answer_text", "question_num"],
    partial_variables={"format_instructions": question_parser.get_format_instructions()},
)

question_chain = question_prompt | llm | question_parser

def extract_text_from_pdf_pages(pdf_path: str, page_numbers: List[int]) -> str:
    """Extract raw text from specified pages (1-indexed)."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    if max(page_numbers) > len(doc):
        doc.close()
        raise ValueError(f"One or more page numbers exceed PDF length ({len(doc)} pages)")

    text_parts = []
    for page_num in sorted(page_numbers):
        page = doc.load_page(page_num - 1)
        page_text = page.get_text("text")
        text_parts.append(f"\n--- Page {page_num} ---\n{page_text}\n")

    doc.close()
    return "\n".join(text_parts).strip()


def extract_single_question(pdf_path: str, page_numbers: List[int], question_num: str) -> QuestionExtraction:
    """Extract structured question using LLM."""
    raw_text = extract_text_from_pdf_pages(pdf_path, page_numbers)

    if not raw_text.strip():
        raise ValueError(f"No text found on pages {page_numbers} in {pdf_path}")

    result = question_chain.invoke({
        "answer_text": raw_text,
        "question_num": question_num
    })

    logger.info(f"Successfully parsed Question {question_num}")
    return result


def save_extracted_data(data: QuestionExtraction, output_path: str) -> None:
    """Save any Pydantic model to JSON with proper formatting."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data.model_dump(exclude_unset=True), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved extracted question → {output_path}")


def extract_and_save_question_answer(
    question_pdf_path: str,
    question_pages: List[int],
    question_num: str,
    output_dir: str = "questions_and_model_answers_json_and_scripts"
) -> str:
    """
    Extract and save only the question (no model answer logic anymore).

    Returns:
        str: Path to the saved JSON file
    """
    try:
        # Extract structured question
        question_data = extract_single_question(question_pdf_path, question_pages, question_num)

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        question_dir = os.path.join(output_dir, "question")
        Path(question_dir).mkdir(parents=True, exist_ok=True)

        output_path = os.path.join(question_dir, f"question_{question_num}_{timestamp}.json")

        # Save
        save_extracted_data(question_data, output_path)

        logger.info(f"Question {question_num} fully extracted and saved")
        return output_path

    except Exception as e:
        logger.error(f"Failed to extract Question {question_num}: {e}")
        raise
