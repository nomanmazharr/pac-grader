import os
from dummy_grade import grade_student
from annotate import annotate_pdf
from data_preprocessing_latest import extract_and_save_question_answer
from logging_config import logger
from marking_criteria_extraction import extract_pdf_annotations_pipeline, merging_marking_criteria

def extract_question_and_model_answer(question_pdf_path, question_pages, model_answer_pdf_path, answer_pages, question_num):
    """Extract question and model answer, return JSON paths."""
    try:
        logger.info(f"Starting extraction for Question {question_num}")
        questions_path, model_answers_path = extract_and_save_question_answer(
            question_pdf_path, question_pages,
            model_answer_pdf_path, answer_pages,
            question_num
        )
        logger.info(f"Extraction completed successfully for Question {question_num}")
        return True, questions_path, model_answers_path

    except Exception as e:
        logger.exception(f"Extraction failed for Question {question_num}: {e}")
        return False, None, None

def extracting_and_merging_marking_criteria(model_answer_pdf_path, answer_pages, model_answers_path):
    """Extract marking criteria from model answers and merge it into the model answers document."""
    try:
        logger.info("Extracting marking criteria...")
        marking_criteria = extract_pdf_annotations_pipeline(model_answer_pdf_path, answer_pages)

        if not marking_criteria:
            logger.warning("Was not able to extract marking criteria")
            return False, model_answers_path

        logger.info("Merging marking criteria into model answers...")
        merged_model_answers_path = merging_marking_criteria(model_answers_path, marking_criteria)

        if not merged_model_answers_path:
            logger.warning("Merging marking criteria failed")
            return False, model_answers_path

        logger.info("Marking criteria successfully extracted and merged.")
        return True, merged_model_answers_path

    except Exception as e:
        print(f"❌ Extraction of marking criteria failed: {e}")
        return False, model_answers_path

def grade_and_annotate_student(student_pdf_path, student_name, questions_path, model_answers_path, question_num, student_pages, output_dir):
    """Grade and annotate student work."""
    try:
        logger.info(f"Starting grading for student '{student_name}' - Question {question_num}")

        # Grade
        grades_csv_path = grade_student(
            student_pdf_path, student_name, questions_path,
            model_answers_path, question_num, student_pages
        )

        if not grades_csv_path:
            logger.warning(f"Grading failed for {student_name} (Question {question_num})")
            return False, "Grading failed", None

        logger.info(f"Grading complete, results saved to CSV for {student_name} (Question {question_num})")

        # Annotate
        logger.info(f"Starting annotation for {student_name} (Question {question_num})")
        success = annotate_pdf(student_pdf_path, output_dir, student_name, grades_csv_path)

        if success:
            annotated_path = os.path.join(output_dir, student_name.lower(), f"{student_name.lower()}_annotated.pdf")
            logger.info(f"Annotation completed successfully for {student_name} (Question {question_num})")
            return True, "Processing completed", annotated_path
        else:
            logger.error(f"Annotation failed for {student_name} (Question {question_num})")
            return False, "Annotation failed", None

    except Exception as e:
        logger.exception(f"Grading/annotation failed for {student_name} (Question {question_num}): {e}")
        return False, str(e), None


def process_exam(
    question_pdf_path, question_pages, model_answer_pdf_path, answer_pages, question_num,
    student_pdf_path, student_pages, output_dir, student_name
):
    """Complete exam processing pipeline."""
    logger.info("=" * 60)
    logger.info(f"📘 Starting processing pipeline for student '{student_name}' (Question {question_num})")
    
    # Step 1: Extract
    extract_success, questions_path, model_answers_path = extract_question_and_model_answer(
        question_pdf_path, question_pages, model_answer_pdf_path, answer_pages, question_num
    )

    if not extract_success:
        logger.error(f"Extraction failed for Question {question_num}")
        return False, "Extraction failed", None, None
    
    # Step 2: Extracting and merging marking criteria
    merging_success, model_answers_path = extracting_and_merging_marking_criteria(model_answer_pdf_path, answer_pages, model_answers_path)

    if not merging_success:
        logger.warning(f'Failed merging final criteria for the Question {question_num}. Using merging criteria based on text')
        return True, "Using textual Marking Criteria ", questions_path, model_answers_path
    
    # Step 3: Grade & Annotate
    grade_success, message, annotated_path = grade_and_annotate_student(
        student_pdf_path, student_name, questions_path, model_answers_path,
        question_num, student_pages, output_dir
    )

    if grade_success:
        logger.info(f"✅ Completed processing for {student_name} (Question {question_num})")
    else:
        logger.warning(f"⚠️ Processing incomplete for {student_name} (Question {question_num}) — {message}")

    logger.info("=" * 60)
    return grade_success, message, questions_path, model_answers_path


# Optional: Direct test entry
if __name__ == "__main__":
    logger.info("Running exam pipeline test mode (no actual grading executed).")
