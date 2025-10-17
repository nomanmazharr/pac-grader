import fitz
import pandas as pd
import ast
import re
import os
from logging_config import logger

def insert_wrapped_text(page, x, y, text, max_width, fontsize, color, fontname, y_limit):
    """Insert wrapped text at (x,y), clipped so it won't cross y_limit."""
    try:
        font = fitz.Font(fontname=fontname)
        words = text.split()
        current_line = ""
        lines = []
        line_height = fontsize + 2

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            width = font.text_length(test_line, fontsize=fontsize)
            if width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        inserted = 0
        for i, line in enumerate(lines):
            y_pos = y + i * line_height
            if y_pos + line_height > y_limit:
                logger.warning(f"Reached y_limit={y_limit:.2f}, truncating comment")
                break
            page.insert_text(
                (x, y_pos),
                line,
                fontsize=fontsize,
                color=color,
                fontname=fontname,
                overlay=True,
            )
            inserted += 1

        logger.info(f"Inserted {inserted}/{len(lines)} wrapped lines at x={x}, y_start={y}, clipped at y_limit={y_limit}")
    except Exception as e:
        logger.error(f"Error inserting wrapped text: {e}")

def insert_tick(page, x0, y0, placed_ticks):
    """Insert tick at the start of a line using the first word's y0."""
    try:
        tick_x = max(10, x0 - 25)
        tick_y = y0 + 10
        tick_key = round(tick_y, 1)

        if tick_key in placed_ticks:
            logger.info(f"  Tick already present near y={tick_y}, skipping...")
            return False

        tw = fitz.TextWriter(page.rect)
        tw.append(
            (tick_x, tick_y),
            chr(0x2714),
            fontsize=12,
        )
        tw.write_text(page, overlay=True)

        placed_ticks.add(tick_key)
        logger.info(f"Inserted tick at ({tick_x}, {tick_y})")
        return True
    except Exception as e:
        logger.error(f"Error inserting tick: {e}")
        return False

def annotate_correct_lines(doc, correct_lines):
    """
    Annotate correct lines with tick marks across the whole document.
    Uses cross-page lookahead before falling back to word chunks.
    Advances page pointer forward (never goes back).
    """
    logger.info("Starting tick annotation for entire document")

    # --- Flatten the question-based stringified lists into a list of actual lines ---
    flat_lines = []
    for idx, entry in enumerate(correct_lines):
        try:
            parsed = ast.literal_eval(entry)
            if isinstance(parsed, list):
                for ln in parsed:
                    if isinstance(ln, str) and ln.strip():
                        flat_lines.append(ln.strip())
            elif isinstance(parsed, str) and parsed.strip():
                flat_lines.append(parsed.strip())
            else:
                # not a list or string we can use
                logger.info(f"Skipping invalid parsed entry at index {idx}: {entry}")
        except (ValueError, SyntaxError):
            # If parsing fails, fall back to using the raw entry if it's a non-empty string
            if isinstance(entry, str) and entry.strip():
                flat_lines.append(entry.strip())
            else:
                logger.info(f"Failed to parse entry at index {idx}, skipping.")

    logger.info(f"Flattened correct_lines: {len(flat_lines)} lines (from {len(correct_lines)} entries)")

    placed_ticks = set()
    line_index = 0
    total_lines = len(flat_lines)
    page_num = 0  # start from first page

    while line_index < total_lines and page_num < len(doc):
        # now take the actual single line to search
        line = flat_lines[line_index]

        search_text = line[:50].strip()
        if not search_text:
            logger.info(f"Skipping empty line at index {line_index}: {line}")
            line_index += 1
            continue

        matched = False

        # --- Try current page
        page = doc[page_num]
        instances = page.search_for(search_text, clip=page.rect)
        if instances:
            logger.info(f"Exact match for '{search_text}' on page {page_num+1}")
            x0, y0, x1, y1 = instances[0]
            line_key = round((y0 + y1) / 2, 1)
            if line_key not in placed_ticks:
                insert_tick(page, x0, y0, placed_ticks)
                placed_ticks.add(line_key)
            matched = True

        # --- Lookahead: search future pages (only if not matched on current page)
        if not matched:
            for next_page in range(page_num+1, len(doc)):
                instances = doc[next_page].search_for(search_text, clip=doc[next_page].rect)
                if instances:
                    logger.info(f"Exact match for '{search_text}' found on future page {next_page+1}")
                    x0, y0, x1, y1 = instances[0]
                    line_key = round((y0 + y1) / 2, 1)
                    if line_key not in placed_ticks:
                        insert_tick(doc[next_page], x0, y0, placed_ticks)
                        placed_ticks.add(line_key)
                    matched = True
                    page_num = next_page  # 🚀 jump forward to this page
                    break


        if not matched:
            for fallback_page in [page_num, page_num + 1]:
                if fallback_page >= len(doc):
                    continue  # no page to check

                fb_page = doc[fallback_page]
                logger.info(f"No exact match for '{search_text}', trying fallback word-by-word on page {fallback_page+1}")

                words_in_line = re.findall(r'\b\w+\b', search_text)
                word_hits = []

                for word in words_in_line:
                    instances = fb_page.search_for(word, clip=fb_page.rect)
                    if instances:
                        word_hits.append(instances[0])

                    if len(word_hits) >= 4:  # ✅ require at least 4 words
                        # Use the Y position of the first hit as anchor
                        logger.info(f"Fallback word match success for '{search_text}' on page {fallback_page+1}")
                        x0, y0, x1, y1 = word_hits[0]
                        line_key = round((y0 + y1) / 2, 1)
                        if line_key not in placed_ticks:
                            insert_tick(fb_page, x0, y0, placed_ticks)
                            placed_ticks.add(line_key)
                        matched = True
                        page_num = fallback_page  # 🚀 update current page if fallback was on next page
                        logger.info(f"Fallback word match success for '{search_text}' on page {fallback_page+1}")
                        break

                if matched:
                    break

        # Move to next flat line
        line_index += 1

        if not matched:
            logger.info(f"Line {line_index} not matched, moving on...")

    logger.info("Completed annotation of all lines.")

        # Don't auto-increment page_num unless explicitly jumped
    logger.info(f"Completed annotation: {line_index}/{total_lines} lines processed.")


def underline_correct_words(page, correct_words, page_num):
    """Underline specific words or phrases from the correct_words list."""
    logger.info(f"Starting underline annotation for page {page_num + 1}")
    
    for word_entry in correct_words:
        try:
            parsed_words = ast.literal_eval(word_entry)
            if not isinstance(parsed_words, list):
                logger.info(f"Skipping invalid word data: {word_entry}")
                continue
        except (ValueError, SyntaxError):
            logger.warning(f"Failed to parse word data: {word_entry}, skipping.")
            continue
        
        for correct_word in parsed_words:
            search_text = correct_word.strip()
            if not search_text:
                logger.info(f"Skipping empty correct word: '{correct_word}'")
                continue
            
            word_instances = page.search_for(search_text, clip=page.rect)
            
            if not word_instances:
                logger.info(f"No exact match for phrase: '{search_text}' on page {page_num}.")
            else:
                logger.info(f"Found exact match for '{search_text}' ({len(word_instances)} instances).")
                for inst in word_instances:
                    try:
                        x0, y0, x1, y1 = inst
                        underline_y = y1 + 2
                        page.draw_line(
                            (x0, underline_y),
                            (x1, underline_y),
                            color=(1, 0, 0),
                            width=1.5,
                            overlay=True
                        )
                        logger.info(f"Underlined '{search_text}' at ({x0}, {underline_y}) to ({x1}, {underline_y}) on page {page_num}.")
                    except Exception as e:
                        logger.error(f"Error underlining '{search_text}': {e}")

    logger.info(f"Completed underline annotation for page {page_num + 1}")

def annotate_pdf(input_dir, output_dir, student_name, grades_csv_path):
    """Annotate PDF with scores, comments, ticks, and underlines."""
    # input_pdf_path = os.path.join(input_dir, f"{student_name}.pdf")
    input_pdf_path = input_dir

    student_lower = student_name.lower()
    output_pdf_path = os.path.join(output_dir, student_lower, f"{student_lower}_annotated.pdf")
    
    logger.info(f"Starting annotation process for {input_pdf_path}")
    
    if not os.path.exists(input_pdf_path):
        logger.error(f"Input PDF not found: {input_pdf_path}")
        return False
    
    if not os.path.exists(grades_csv_path):
        logger.error(f"Grades CSV not found: {grades_csv_path}")
        return False
    
    try:
        logger.info(f"Loading grades from {grades_csv_path}")
        grades_df = pd.read_csv(grades_csv_path)
        logger.info(f"Loaded {len(grades_df)} grading records")
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {grades_csv_path}")
        return False
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return False

    correct_lines = grades_df['correct_lines'].dropna().tolist()
    correct_words = grades_df['correct_words'].dropna().tolist()
    logger.info(f"Loaded {len(correct_lines)} correct lines and {len(correct_words)} correct words from CSV")
    
    try:
        logger.info(f"Opening PDF: {input_pdf_path}")
        doc = fitz.open(input_pdf_path)
        logger.info(f"PDF opened with {len(doc)} pages")
    except Exception as e:
        logger.error(f"Error opening PDF: {e}")
        return False
    
    try:
        logger.info("Creating score dictionary from grades")
        score_dict = {str(row['question_number']): f"{row['score']}/{row['total_marks']}" for _, row in grades_df.iterrows()}
        logger.info(f"Score dictionary created with {len(score_dict)} entries")
        
        comment_dict = {str(row['question_number']): row.get('comment', '') for _, row in grades_df.iterrows()}
        
        for page_num in range(len(doc)):
            logger.info(f"Processing page {page_num + 1}")
            page = doc[page_num]
            
            logger.info("Extracting text from page")
            text_on_page = page.get_text("text")
            logger.info(f"Extracted {len(text_on_page)} characters of text")
            
            text_blocks = page.get_text("dict")["blocks"]
            
            logger.info("Searching for question numbers")
            question_pattern = r'(?<!\d)\d+\.\d+(?:\([a-z]\))?(?!\d)'
            question_matches = list(re.finditer(question_pattern, text_on_page))
            logger.info(f"Found {len(question_matches)} potential question matches")
            
            question_positions = []
            for m in question_matches:
                q_num = m.group(0)
                rects = page.search_for(q_num)
                if rects:
                    y0 = rects[0].y0
                    start_pos = m.start()
                    question_positions.append((q_num, y0, start_pos))
                    logger.info(f"Matched {q_num} at y={y0:.2f}")
            
            question_positions.sort(key=lambda x: x[1])
            
            annotated_questions = set()
            
            for i, (q_num, y0, start_pos) in enumerate(question_positions):
                y1 = (question_positions[i+1][1] 
                      if i+1 < len(question_positions) else page.rect.height)
                
                line_start = text_on_page.rfind('\n', 0, start_pos) + 1
                line_end = text_on_page.find('\n', start_pos)
                if line_end == -1:
                    line_end = len(text_on_page)
                surrounding_text = text_on_page[line_start:line_end].strip()
                logger.info(f"Surrounding text on same line: '{surrounding_text}'")
                
                if any(keyword in surrounding_text.lower() for keyword in ['marks', '/', 'score', 'total']):
                    logger.info(f"Skipping {q_num} as it appears to be a score")
                else:
                    if q_num in score_dict and q_num not in annotated_questions:
                        logger.info(f"Annotating {q_num} with score {score_dict[q_num]}")
                        text_instances = page.search_for(q_num)
                        if text_instances:
                            inst = text_instances[0]
                            
                            x_offset = -40
                            text_x = inst.x0 + x_offset
                            text_y = inst.y0 + 10
                            
                            score_text = score_dict[q_num]
                            page.insert_text(
                                (text_x, text_y),
                                score_text,
                                fontsize=12,
                                color=(0, 0, 1)
                            )
                            
                            tw = fitz.TextWriter(page.rect)
                            tw.append((text_x, text_y), score_text)
                            text_rect = tw.rect
                            tw = None
                            
                            annotated_questions.add(q_num)
                            logger.info(f"Successfully annotated {q_num} on page {page_num + 1}")
                        else:
                            logger.warning(f"Could not find position for {q_num} on page {page_num + 1}")
                    else:
                        logger.info(f"Question {q_num} not in grades or already annotated")
                
                if q_num in comment_dict:
                    comment = comment_dict[q_num]
                    logger.info(f"Annotating {q_num} between y={y0:.2f} and y={y1:.2f} with comment: {comment}")
                    x_left = page.rect.width - 90
                    max_width = 90
                    insert_wrapped_text(
                        page,
                        x_left,
                        y0,
                        comment,
                        max_width=max_width,
                        fontsize=8,
                        color=(1, 0, 0),
                        fontname="helv",
                        y_limit=y1 - 5
                    )
            
            
            if correct_words:
                underline_correct_words(page, correct_words, page_num)
        
        if correct_lines:
                annotate_correct_lines(doc, correct_lines)

        logger.info(f"Saving annotated PDF to {output_pdf_path}")
        os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
        doc.save(output_pdf_path)
        logger.info(f"Annotation process completed. Saved as {output_pdf_path}")
        return True
    except Exception as e:
        logger.error(f"Error during annotation: {e}")
        return False
    finally:
        doc.close()