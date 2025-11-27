import fitz
import pandas as pd
import ast
import re
import os
from logging_config import logger

def underline_correct_words(page, correct_words, page_num):
    logger.info(f"Starting underline annotation → Page {page_num + 1}")
    
    total_underlined = 0
    for word_entry in correct_words:
        try:
            parsed_words = ast.literal_eval(word_entry)
            if not isinstance(parsed_words, list):
                logger.debug(f"Skipping invalid word data (not list): {word_entry}")
                continue
        except (ValueError, SyntaxError) as e:
            logger.debug(f"Failed to parse word entry: {word_entry} | Error: {e}")
            continue
        
        for correct_word in parsed_words:
            search_text = correct_word.strip()
            if not search_text:
                continue
            
            word_instances = page.search_for(search_text, clip=page.rect)
            
            if not word_instances:
                logger.debug(f"Not found on page {page_num + 1}: '{search_text}'")
            else:
                logger.info(f"Underlined: '{search_text}' ({len(word_instances)}×) → Page {page_num + 1}")
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
                        total_underlined += 1
                    except Exception as e:
                        logger.error(f"Error drawing underline for '{search_text}': {e}")

    logger.info(f"Completed underlines → Page {page_num + 1} | Total: {total_underlined}")


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

def create_line_annotator(correct_lines):
    flat_lines = []
    for entry in correct_lines:
        try:
            parsed = ast.literal_eval(entry)
            if isinstance(parsed, list):
                flat_lines.extend([ln.strip() for ln in parsed if ln.strip()])
            elif isinstance(parsed, str) and parsed.strip():
                flat_lines.append(parsed.strip())
        except:
            if isinstance(entry, str) and entry.strip():
                flat_lines.append(entry.strip())

    logger.info(f"Line annotator ready → {len(flat_lines)} correct lines loaded")

    line_index = 0
    placed_ticks = set()

    def annotate_on_page_and_beyond(page, page_num, doc):
        nonlocal line_index
        ticks_this_page = 0

        while line_index < len(flat_lines):
            line = flat_lines[line_index]
            search_text = line[:50].strip()
            if not search_text:
                line_index += 1
                continue

            matched = False

            # Try current page
            if page.search_for(search_text):
                instances = page.search_for(search_text)
                x0, y0, _, _ = instances[0]
                if insert_tick(page, x0, y0, placed_ticks):
                    ticks_this_page += 1
                matched = True
                line_index += 1
                continue

            # Look ahead
            for pn in range(page_num + 1, len(doc)):
                if doc[pn].search_for(search_text):
                    inst = doc[pn].search_for(search_text)[0]
                    if insert_tick(doc[pn], inst.x0, inst.y0, placed_ticks):
                        ticks_this_page += 1
                    line_index += 1
                    matched = True
                    break

            if not matched:
                # Fallback word matching (simplified)
                words = re.findall(r'\b\w+\b', search_text)
                for pn in [page_num, page_num + 1]:
                    if pn >= len(doc):
                        break
                    test_page = doc[pn]
                    hits = [test_page.search_for(w) for w in words if test_page.search_for(w)]
                    if len(hits) >= 4:
                        x0, y0 = hits[0][0].x0, hits[0][0].y0
                        if insert_tick(test_page, x0, y0, placed_ticks):
                            ticks_this_page += 1
                        line_index += 1
                        matched = True
                        break

            if not matched:
                logger.debug(f"Line not found: {search_text}")
                line_index += 1

        logger.info(f"Page {page_num + 1}: {ticks_this_page} ticks placed | Progress: {line_index}/{len(flat_lines)}")
        return line_index >= len(flat_lines)

    return annotate_on_page_and_beyond


def insert_wrapped_text(page, x, y, text, max_width, fontsize, color, fontname, y_limit):
    if not text.strip():
        return
    try:
        lines = []
        words = text.split()
        current = ""
        for word in words:
            test = current + (" " if current else "") + word
            if fitz.get_text_length(test, fontsize=fontsize) <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        for i, line in enumerate(lines):
            y_pos = y + i * (fontsize + 2)
            if y_pos + fontsize > y_limit:
                logger.debug(f"Comment truncated at y_limit={y_limit:.1f}")
                break
            page.insert_text((x, y_pos), line, fontsize=fontsize, color=color,
                             fontname=fontname, overlay=True)
        logger.info(f"Comment inserted → {len(lines)} lines at x={x}")
    except Exception as e:
        logger.error(f"Failed to insert wrapped text: {e}")


def annotate_comments_and_scores(page, page_num, doc, score_dict, comment_dict):
    logger.info(f"Adding scores & comments → Page {page_num + 1}")
    text = page.get_text("text")
    if not text.strip():
        logger.info("No text on page, skipping comments")
        return

    matches = list(re.finditer(r'(?<!\d)\d+\.\d+(?:\([a-z]\))?(?!\d)', text))
    if not matches:
        logger.info("No question numbers found")
        return

    questions = []
    for m in matches:
        q = m.group(0)
        if page.search_for(q):
            questions.append((q, page.search_for(q)[0].y0))

    questions.sort(key=lambda x: x[1])

    for i, (q_num, y0) in enumerate(questions):
        y1 = questions[i+1][1] if i+1 < len(questions) else page.rect.height
        nearby = page.get_text("text", clip=fitz.Rect(0,y0-30,page.rect.width,y0+30))
        # if any(k in nearby.lower() for k in ['total', 'marks', 'score', '/']):
        #     continue

        # Score
        if q_num in score_dict:
            score_text = score_dict[q_num]
            page.insert_text((page.search_for(q_num)[0].x0 - 40, y0 + 8),
                             score_text, fontsize=12, color=(0,0,1))
            logger.info(f"Score added: {q_num} → {score_text}")

        # Comment
        if q_num in comment_dict:
            comment = str(comment_dict[q_num]).strip()
            if comment and comment.lower() not in ['nan', 'none', '']:
                insert_wrapped_text(page, page.rect.width - 95, y0 + 5, comment,
                                    max_width=90, fontsize=8, color=(1,0,0),
                                    fontname="helv", y_limit=y1 - 10)
                logger.info(f"Comment added for Q{q_num}")


def annotate_pdf(input_dir, output_dir, student_name, grades_csv_path, student_pages=None):
    logger.info(f"Starting annotation for: {student_name}")
    input_pdf_path = input_dir
    student_lower = student_name.lower()
    output_pdf_path = os.path.join(output_dir, student_lower, f"{student_lower}_annotated.pdf")
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

    if not os.path.exists(input_pdf_path):
        logger.error(f"PDF not found: {input_pdf_path}")
        return False
    if not os.path.exists(grades_csv_path):
        logger.error(f"CSV not found: {grades_csv_path}")
        return False

    try:
        grades_df = pd.read_csv(grades_csv_path)
        logger.info(f"Loaded {len(grades_df)} grading records")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return False

    correct_lines = grades_df['correct_lines'].dropna().tolist()
    correct_words = grades_df['correct_words'].dropna().tolist()
    logger.info(f"Loaded {len(correct_lines)} correct lines, {len(correct_words)} correct words")

    score_dict = {str(r['question_number']): f"{r['score']}/{r['total_marks']}" for _, r in grades_df.iterrows()}
    comment_dict = {str(r['question_number']): r.get('comment', '') for _, r in grades_df.iterrows()}

    try:
        doc = fitz.open(input_pdf_path)
        logger.info(f"PDF opened → {len(doc)} pages")
    except Exception as e:
        logger.error(f"Cannot open PDF: {e}")
        return False

    pages_to_process = range(len(doc)) if student_pages is None else [p-1 for p in student_pages if 1 <= p <= len(doc)]
    annotate_lines_progressively = create_line_annotator(correct_lines)

    for page_num in pages_to_process:
        logger.info(f"Processing page {page_num + 1}/{len(doc)}")
        page = doc[page_num]

        if correct_words:
            underline_correct_words(page, correct_words, page_num)

        if correct_lines:
            done = annotate_lines_progressively(page, page_num, doc)
            if done:
                logger.info("All correct lines annotated early")
                # optionally break

        annotate_comments_and_scores(page, page_num, doc, score_dict, comment_dict)

    try:
        doc.save(output_pdf_path)
        logger.info(f"SUCCESS → Annotated PDF saved: {output_pdf_path}")
        doc.close()
        return True
    except Exception as e:
        logger.critical(f"Failed to save PDF: {e}")
        doc.close()
        return False