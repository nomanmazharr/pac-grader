import io
import fitz
from llm_setup import client
import json
from typing import Dict, List
from logging_config import logger

def subset_pdf_bytes(pdf_path: str, pages: list[int]) -> io.BytesIO:
    """
    Extract specified pages from a PDF and return as in-memory BytesIO PDF.
    Ensures the buffer behaves like a real PDF file.
    """
    doc = fitz.open(pdf_path)
    new_doc = fitz.open()

    for p in pages:
        if 1 <= p <= len(doc):
            new_doc.insert_pdf(doc, from_page=p-1, to_page=p-1)

    # convert to proper PDF bytes
    pdf_bytes = new_doc.tobytes()
    buf = io.BytesIO(pdf_bytes)
    buf.name = "subset.pdf"   # this is critical for OpenAI to detect it's a PDF
    buf.seek(0)

    doc.close()
    new_doc.close()
    return buf

def upload_pdf_to_openai(pdf_buffer: io.BytesIO, filename: str = "subset.pdf"):
    """
    Upload an in-memory PDF to OpenAI and return file object.
    """
    file_obj = client.files.create(
        file=pdf_buffer,
        purpose="user_data"
        # filename=filename
    )
    return file_obj

annotation_schema = {
    "type": "object",
    "properties": {
        "annotations": {
            "type": "array",
            "description": "List of extracted annotations. Each item maps a question to a heading and marking criteria.",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question identifier, e.g 1.1, 3.1 with subsections if present"
                    },
                    "heading": {
                        "type": "string",
                        "description": "Local heading/section where the annotation appears."
                    },
                    "marking_criteria": {
                        "type": "string",
                        "description": "The red handwritten annotation text (extracted)."
                    }
                },
                "required": ["question", "heading", "marking_criteria"],
                "additionalProperties": False
            }
        }
    },
    "required": ["annotations"],
    "additionalProperties": False
}


def extract_annotations_with_llm(file_obj):
    """
    Ask the LLM to extract red handwritten annotations and map them to question numbers and headings.
    Returns structured JSON.
    """
    response = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_id": file_obj.id,
                    },
                    {
                        "type": "input_text",
                        "text": (
                        """You are analyzing an annotated exam paper (PDF format).
                        Your task is to extract all *red handwritten annotations* visible in the document.

                        For each annotation:
                        - Identify the **question number** it corresponds to.
                        - Determine the **relevant heading or subheading** under which the annotation appears.
                            (The heading should be specific enough to clearly indicate which part or point the annotation refers to — 
                            use the most relevant local heading rather than only the main section title.)
                        - Associate the annotation text directly with the corresponding answer content.

                        Return the output as structured JSON following the provided schema.
                        Ensure all extracted annotations are correctly mapped under their respective question numbers and headings.
                        """
                    ),
                    },
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "annotation_extraction",
                "strict": True,
                "schema": annotation_schema,
                "description": "Schema: annotations array with question, heading and marking_criteria"
            }
        }
    )

    try:
        return json.loads(response.output_text)
    except json.JSONDecodeError as e:
        logger.error(f"⚠️ Failed to parse LLM output as JSON: {e}")
        logger.error(f"Raw response: {response.output_text[:500]}...")  # print first 500 chars for debug
        return None
    
def extract_pdf_annotations_pipeline(pdf_path: str, pages: list[int]):
    """
    Full end-to-end pipeline:
      1. Extract subset of pages (in memory)
      2. Upload to OpenAI
      3. Get annotations JSON
    """
    logger.info(f"Extracting pages {pages} from {pdf_path}...")
    subset_buf = subset_pdf_bytes(pdf_path, pages)

    logger.info("Uploading subset to OpenAI...")
    file_obj = upload_pdf_to_openai(subset_buf)

    logger.info("Extracting annotations using LLM...")
    result_json = extract_annotations_with_llm(file_obj)

    logger.info("✅ Extraction complete.")
    return result_json


def merging_marking_criteria(model_answer_path: str, extracted_data: dict) -> str:
    """
    Merge extracted handwritten annotations (marking criteria) into the main question structure,
    save the updated JSON back to the same file, and return the file path.

    Args:
        model_answer_path (str): Path to the main question JSON file.
        extracted_data (dict): Parsed JSON containing 'annotations' array.

    Returns:
        str: Path to the updated JSON file.
    """

    # Load main question data
    with open(model_answer_path, "r", encoding="utf-8") as f:
        main_data = json.load(f)

    # Build a map: question_number -> list of extracted annotation items
    extracted_map: Dict[str, List[dict]] = {}
    for ann in extracted_data.get("annotations", []):
        q = ann.get("question")
        if q:
            extracted_map.setdefault(q.strip(), []).append({
                "heading": ann.get("heading", "").strip(),
                "marking_criteria": ann.get("marking_criteria", "").strip(),
            })

    # Recursive helper to merge annotations into answers/sub_answers
    def merge_into_answers(answers: List[dict]):
        for ans in answers:
            qnum = ans.get("question_number", "").strip()
            if qnum in extracted_map:
                ans["annotations"] = extracted_map[qnum]

            # Go deeper into nested sub-answers if any
            if ans.get("sub_answers"):
                merge_into_answers(ans["sub_answers"])

    # Merge into the main structure
    if "answers" in main_data:
        merge_into_answers(main_data["answers"])

    # Save the updated JSON back to file
    with open(model_answer_path, "w", encoding="utf-8") as f:
        json.dump(main_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Merged annotations saved to: {model_answer_path}")
    return model_answer_path


# # Example usage:
# if __name__ == "__main__":
#     # Load your two JSONs from file or variables
#     with open("first_model_answer_2025-11-04_09-52-22.json") as f:
#         main_questions = json.load(f)
    
#     # with open("extracted_annotations.json") as f:
#     #     extracted_annotations = json.load(f)
    
#     merged_output = merging_marking_criteria(main_questions, extracted_annotations)
    
#     with open("first_model_answer_2025-11-04_09-52-22.json", "w", encoding='utf-8') as f:
#         json.dump(merged_output, f, indent=2, ensure_ascii=False)
    
#     print("✅ Merged annotations successfully added to question structure.")
