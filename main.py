import os
from typing import Tuple, Optional
from logging_config import logger
from data_preprocessing import extract_and_save_question_answer
from rubric_extraction import extract_pdf_annotations_pipeline
from grade import grade_student
from annotate import annotate_pdf


def extract_question_text(
    question_pdf_path: str,
    question_pages: list,
    question_num: str
) -> Tuple[bool, Optional[str]]:
    """Extract question text from the exam paper."""
    try:
        logger.info(f"Extracting Question {question_num} from pages {question_pages}")
        questions_path = extract_and_save_question_answer(
            question_pdf_path, question_pages, question_num
        )

        if not questions_path or not os.path.isfile(questions_path):
            logger.error(f"Question {question_num} extraction failed – file not created")
            return False, None

        logger.info(f"Question {question_num} extracted → {os.path.basename(questions_path)}")
        return True, questions_path

    except Exception as e:
        logger.error(f"Failed to extract Question {question_num}: {e}")
        logger.debug(f"Traceback: {e.__traceback__}")
        return False, None


def extract_marking_rubric(
    model_answer_pdf_path: str,
    answer_pages: list
) -> Tuple[bool, Optional[str]]:
    """Try to extract annotation-based rubric from model answer."""
    try:
        logger.info(f"Attempting to extract marking rubric from model answer (pages {answer_pages})")
        rubric_path = extract_pdf_annotations_pipeline(model_answer_pdf_path, answer_pages)

        if not rubric_path or not os.path.isfile(rubric_path):
            logger.warning("No annotation-based rubric found. Will use text-based grading fallback.")
            return False, None

        logger.info(f"Rubric extracted successfully → {os.path.basename(rubric_path)}")
        return True, rubric_path

    except Exception as e:
        logger.error(f"Rubric extraction failed: {e}")
        logger.debug(f"Traceback: {e.__traceback__}")
        return False, None


def grade_and_annotate_student(
    student_pdf_path: str,
    student_name: str,
    questions_path: str,
    model_answers_path: Optional[str],
    question_num: str,
    student_pages: list,
    output_dir: str
) -> Tuple[bool, str, Optional[str]]:
    """Grade student + generate annotated PDF."""
    try:
        # Ensure output directories exist
        student_dir = os.path.join(output_dir, student_name.lower())
        os.makedirs(student_dir, exist_ok=True)

        logger.info(f"Grading {student_name} – Question {question_num}")

        # === 1. Grading ===
        grades_csv_path = grade_student(
            student_pdf_path=student_pdf_path,
            student_name=student_name,
            questions_path=questions_path,
            model_answers_path=model_answers_path,   # Can be None → your grader handles it
            question_number=question_num,
            student_pages=student_pages
        )

        if not grades_csv_path or not os.path.isfile(grades_csv_path):
            logger.error(f"Grading failed for {student_name} (Q{question_num})")
            return False, "Grading failed", None

        logger.info(f"Grading complete → {os.path.basename(grades_csv_path)}")

        # === 2. Annotation ===
        logger.info(f"Creating annotated PDF for {student_name}...")
        annotation_ok = annotate_pdf(student_pdf_path, output_dir, student_name, grades_csv_path, student_pages)

        annotated_pdf = os.path.join(student_dir, f"{student_name.lower()}_annotated.pdf")

        if annotation_ok and os.path.isfile(annotated_pdf):
            logger.info(f"Annotated PDF ready → {os.path.basename(annotated_pdf)}")
            return True, "Completed successfully", annotated_pdf
        else:
            logger.error(f"Annotation failed for {student_name}")
            return False, "Annotation failed", None

    except Exception as e:
        logger.error(f"Unexpected error for {student_name} (Q{question_num}): {e}")
        logger.debug(f"Traceback: {e.__traceback__}")
        return False, f"Exception: {e}", None


def process_exam(
    question_pdf_path: str,
    question_pages: list,
    question_num: str,
    model_answer_pdf_path: str,
    answer_pages: list,
    student_pdf_path: str,
    student_pages: list,
    student_name: str,
    output_dir: str
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """Full pipeline – one student, one question."""
    logger.info("=" * 70)
    logger.info(f"PIPELINE START → {student_name} | Question {question_num}")
    logger.info("=" * 70)

    # Step 1: Extract question
    q_ok, questions_path = extract_question_text(question_pdf_path, question_pages, question_num)
    if not q_ok:
        logger.error("Pipeline stopped: Question extraction failed")
        return False, "Question extraction failed", None, None

    # Step 2: Try to get rubric (optional)
    rubric_ok, model_answers_path = extract_marking_rubric(model_answer_pdf_path, answer_pages)
    if not rubric_ok:
        model_answers_path = None  # Let grader fall back to text-based marking

    # Step 3: Grade + Annotate
    success, message, annotated_path = grade_and_annotate_student(
        student_pdf_path=student_pdf_path,
        student_name=student_name,
        questions_path=questions_path,
        model_answers_path=model_answers_path,
        question_num=question_num,
        student_pages=student_pages,
        output_dir=output_dir
    )

    status = "SUCCESS" if success else "FAILED"
    logger.info(f"PIPELINE {status} → {student_name} (Q{question_num}) | {message}")
    logger.info("=" * 70 + "\n")

    return success, message, questions_path, annotated_path

# if __name__ == "__main__":
#     logger.info("Exam Grading Pipeline – Test Run")

#     process_exam(
#         question_pdf_path='questions_and_models_answers_pdf/testing/orchid/ICPR01(ME4)DEC24_Qs - Orchid_d2 viewing.pdf',
#         question_pages=[9, 10],
#         question_num='4',
#         model_answer_pdf_path='questions_and_models_answers_pdf/testing/orchid/ICAEW_2024_AA_Mock_Orchid_As_GAP_marked up (2024).pdf',
#         answer_pages=[19, 20, 21, 22],
#         student_pdf_path='questions_and_models_answers_pdf/testing/orchid/Aaishah Sedoo_548547_assignsubmission_file_AA_GE_AAISHAH_SEDOO.pdf',
#         student_pages=[6, 7],
#         student_name='Demilade',
#         output_dir='annotation_nov25'
#     )