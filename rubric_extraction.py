import io
import fitz
from llm_setup import client
import json
from logging_config import logger

def _create_pdf_subset(pdf_path: str, pages: list[int]) -> io.BytesIO:
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

def _upload_to_openai(pdf_buffer: io.BytesIO, filename: str = "subset.pdf"):
    """
    Upload an in-memory PDF to OpenAI and return file object.
    """
    file_obj = client.files.create(
        file=pdf_buffer,
        purpose="user_data"
    )
    logger.info(f"Uploaded → file_obj: {file_obj}")
    return file_obj

model_answer_schema = {
    "type": "object",
    "properties": {
        "question_title": {
            "type": "string",
            "description": "Main question title, e.g., 'Question 4'"
        },
        "description": {
            "type": ["string", "null"],
            "description": "Introductory paragraph or assumptions if present"
        },
        "total_marks": {
            "type": ["string", "null"],
            "description": "Total marks for the main question"
        },
        "answers": {
            "type": "array",
            "description": "Model answer if no subsections are present for this question",
            "items": {  # top-level answer object
                "type": "object",
                "properties": {
                    "question_number": {
                        "type": "string",
                        "description": "Subquestion number such as '4.1', '4.1(a)', etc. if question number explicitly present don't include heading into question number else if subsections are on base of heading only use heading in that case"
                    },
                    "answer": {
                        "type": ["string", "null"],
                        "description": "Model answer content for this question_number, never include marking criteria in answer"
                    },
                    "marking_criteria": {
                        "type": ["string", "null"],
                        "description": "Both typed marking criteria AND handwritten annotations merged here for the questions if possible headings, don't include answer in the marking criteria"
                    },
                    "total_marks_available": {
                        "type": ["string", "null"],
                        "description": "Marks available for this specific part"
                    },
                    "maximum_marks": {
                        "type": ["string", "null"],
                        "description": "Maximum marks if explicitly mentioned"
                    },
                    "sub_answers": {
                        "type": ["array", "null"],
                        "description": "Nested subdivisions (e.g. (a), (b), (i), etc.)",
                        "items": {  # recursive inline definition
                            "type": "object",
                            "properties": {
                                "question_number": {"type": "string", "description": "Sub-subquestion number, don't include heading if question number explicitly present for sub answer"},
                                "answer": {"type": ["string", "null"], "description": "Model answer content, don't include marking criteria in answers "},
                                "marking_criteria": {"type": ["string", "null"], "description": "Merged marking criteria + handwritten notes, no answer in marking criteria"},
                                "total_marks_available": {"type": ["string", "null"], "description": "Marks available"},
                                "maximum_marks": {"type": ["string", "null"], "description": "Maximum marks"},
                                "sub_answers": {  # can be nested further
                                    "type": ["array", "null"],
                                    "description": "Further nested subdivisions",
                                    "items": { "type": "object", "properties": {}, "required": [], "additionalProperties": False }
                                }
                            },
                            "required": ["question_number", "answer", "marking_criteria", "total_marks_available", "maximum_marks", "sub_answers"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["question_number", "answer", "marking_criteria", "total_marks_available", "maximum_marks", "sub_answers"],
                "additionalProperties": False
            }
        }
    },
    "required": ["question_title", "answers", "description", "total_marks"],
    "additionalProperties": False
}


def _extract_rubric_with_vision(file_obj):
    response = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_obj.id},
                    {"type": "input_text", "text": """
You are an expert in extracting and structuring model answers and marking criteria from exam marking guides, including any handwritten annotations visible in the PDF.

Focus strictly on Question that is present as a whole and extact all of its model answers, printed marking criteria, and handwritten annotations.

You are provided a PDF file containing:

- Model answers
- Printed marking criteria
- Headings and subheadings
- Handwritten annotations (red ink)

Your task is to extract all content directly from the PDF in one step and return structured JSON that strictly follows the provided schema.

IMPORTANT GLOBAL RULES:
- Preserve wording EXACTLY as written. No rewriting, rephrasing, summarizing, or adding interpretations.
- NEVER omit marking criteria for any subsection if marking criteria text exists anywhere in the PDF.
- ALWAYS extract marking criteria for the main question AND for each subsection separately if present.
- Handwritten annotations must be merged directly into the `marking_criteria` of the corresponding question or subsection.
- If marking criteria applies to multiple subsections or the entire main question, you MUST replicate it across all relevant subsections.
- If the model is unsure where a criteria block or annotation belongs, assign it to the main question AND to all subsections to avoid any loss of information.

---

## EXTRACTION RULES

### 1. Identify Subsections Reliably
A subsection MUST be created whenever explicit labels appear, such as:
- Numeric: 1.1, 1.2, 1.3 …
- Alphabetic: a), b), c)
- Roman or nested: (i), (ii), (iii)

If subsections exist:
- Create a separate entry for **each subsection**.
- Each subsection must contain:
  - `question_number`
  - `answer`
  - `marking_criteria` (printed + handwritten)

If NO subsections exist:
- Create only one entry, using the main question number.

---

### 2. Marking Criteria and Annotations Handling (STRICT)
For EVERY question and subsection:
- Extract ALL marking criteria and handwritten annotation text that applies to that part.
- The `marking_criteria` field must contain only marking instructions (no maximum_marks or total_marks_available).
- If marking criteria itself contains mark allocations for sub-elements (e.g., “explanation - 2 marks”), keep them exactly as written.
- Handwritten annotations must be integrated directly into the `marking_criteria` of the corresponding question or subsection.
- Never write explicitly word "Handwritten Annotations" in marking criteria when you are extracting red handwritten marking criteria.
- Do NOT create a separate annotations array.
- In case of marking guide isn't available marking criteria is mostly given in the Maximum Marks section only get the criteria from there not the maximum marks.

---

### 2.1 Marking Criteria and Annotations Assignment Logic (VERY IMPORTANT)
- If marking criteria or annotations appear BEFORE any subsection → treat them as criteria for the **main question**.
- If they appear AFTER a subsection → assign them to **that subsection**.
- If one block applies to MULTIPLE subsections → COPY the entire block to each subsection.
- If the model cannot confidently determine which subsection a criteria or annotation block belongs to → assign that block to the **main question AND to all subsections**.
- NEVER leave a subsection without marking criteria if any criteria exists that could apply to it.
- NEVER includ instructions in the marking criteria like ["red handwritten notes", "red ticks"], marking criteria is available in marking guide or with the maximum marks in text and in annotations if red notes are present include them to the question they belongs to never add instructions by yourself.

---

### 3. Marks Extraction
- Extract **maximum_marks** using phrases like: "Maximum marks", "Maximum full marks", "Maximum"
- Extract **total_marks_available** using phrases like: "Total Possible Marks", "Marks Available", "Total Marks"
- NEVER leave maximum_marks or total_marks_available.
- If a subsection has nested sub-subsections, maximum marks and total available marks appear ONLY once for the parent subsection, never include them in further subsections only once in parent.

---

### 4. Answer vs Marking Criteria Separation
- `answer` contains ONLY the model answer content.
- `marking_criteria` contains ONLY marking rules (printed + handwritten).
- DO NOT include maximum marks or total marks inside `marking_criteria`.

---

### 5. Hierarchy Enforcement
Always preserve the correct structure:

- Main question  
  → Subquestion (1.1, 1.2)  
    → Nested sub ((a), (b))  
      → Deeper nested ((i), (ii))

Use `sub_answers` ONLY when nested subsections exist beneath the immediate subsection.

---

Return only valid JSON — no markdown, no commentary, no preamble.
"""}
                ]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "universal_exam_rubric",
                "strict": True,
                "schema": model_answer_schema  # this is your updated flexible schema
            }
        }
    )

    try:
        raw_json = response.output_text.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:-3].strip()
        data = json.loads(raw_json)
        logger.info("Rubric successfully parsed from vision model")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from model response: {e}")
        logger.debug(f"Raw output: {raw_json[:500]}...")
        raise

from datetime import datetime
from pathlib import Path
def extract_pdf_annotations_pipeline(pdf_path: str, pages: list[int], output_dir= "questions_and_model_answers_json_and_scripts"):
    """
    Full end-to-end pipeline:
      1. Extract subset of pages (in memory)
      2. Upload to OpenAI
      3. Get annotations JSON
    """
    try:
        logger.info(f"Starting rubric extraction from pages {pages}")

        # 1. Create subset
        pdf_buffer = _create_pdf_subset(pdf_path, pages)

        # 2. Upload
        file_id = _upload_to_openai(pdf_buffer)

        # 3. Extract with vision LLM
        rubric_data = _extract_rubric_with_vision(file_id)

        # 4. Save to disk
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rubric_dir = Path(output_dir) / "rubric"
        rubric_dir.mkdir(parents=True, exist_ok=True)

        output_path = rubric_dir / f"rubric_extracted_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rubric_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Rubric saved → {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Rubric extraction pipeline failed: {e}")
        logger.debug(f"Traceback: {e}", exc_info=True)
        return None