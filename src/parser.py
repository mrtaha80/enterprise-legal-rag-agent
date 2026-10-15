import re
import pymupdf as fitz
from typing import List, Dict
from src.security import SecurityGuard
from src.logger import logger

CLAUSE_PATTERNS = [
    r"^(?:ماده|بند|تبصره|فصل|بخش|پیوست|صورتجلسه)\s*[\d\u0660-\u0669\u06F0-\u06F9IVXLCDM]+[:\.\-–]",
    r"^(?:Article|Section|Clause|Schedule|Exhibit)\s*[\dIVXLCDM]+[:\.\-–]",
    r"^[\(\（]?\s*[\d\u0660-\u0669\u06F0-\u06F9]+\s*[\)\）\.\-–]",
    r"^[الفبپتثجحخدذرزژسشصضطظعغفقکگلمنوهی]\s*[\.\-–\)]",
]

COMBINED_PATTERN = re.compile("|".join(CLAUSE_PATTERNS), re.IGNORECASE | re.MULTILINE)


def _split_into_clauses(text: str) -> List[str]:
    """تفکیک متن صفحه بر اساس ساختار مواد و بندهای قانونی"""
    splits = [m.start() for m in COMBINED_PATTERN.finditer(text)]
    if not splits:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs if paragraphs else [text.strip()]

    clauses = []
    if splits[0] > 0:
        preamble = text[:splits[0]].strip()
        if preamble:
            clauses.append(preamble)

    for i in range(len(splits)):
        start = splits[i]
        end = splits[i + 1] if i + 1 < len(splits) else len(text)
        clause = text[start:end].strip()
        if clause:
            clauses.append(clause)

    return clauses


def _subdivide_long_clause(clause: str, max_chars: int = 1200, overlap_words: int = 25) -> List[str]:
    """
    خردایش ایمن بندهای طولانی بر اساس پنجره لغزان کلمات (Tail-Safe Sliding Window).
    تضمین می‌کند که هیچ بخش انتهایی (Tail) از دست نرود و چانک تکراری خالص تولید نشود.
    """
    clause = clause.strip()
    if not clause:
        return []
    if len(clause) <= max_chars:
        return [clause]

    words = clause.split()
    sub_chunks = []
    start_idx = 0
    total_words = len(words)

    while start_idx < total_words:
        current_words = []
        current_len = 0
        end_idx = start_idx

        while end_idx < total_words:
            word = words[end_idx]
            added_len = len(word) if current_len == 0 else len(word) + 1

            if current_len + added_len > max_chars and current_words:
                break

            current_words.append(word)
            current_len += added_len
            end_idx += 1

        chunk_str = " ".join(current_words).strip()
        if chunk_str:
            sub_chunks.append(chunk_str)

        if end_idx >= total_words:
            break

        next_start = end_idx - overlap_words
        if next_start <= start_idx:
            next_start = start_idx + 1
        start_idx = next_start

    return sub_chunks


def extract_chunks_from_pdf(file_path: str, source_name: str = None) -> List[Dict]:
    """استخراج و تفکیک امن اسناد PDF با مرتب‌سازی هندسی بلوک‌ها و حذف چانک‌های آلوده"""
    doc = fitz.open(file_path)
    filename = source_name if source_name else file_path.split("\\")[-1].split("/")[-1]
    all_chunks = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks", sort=True)
            page_text = "\n".join([b[4].strip() for b in blocks if len(b) > 4 and b[4].strip()])

            if not page_text:
                continue

            raw_clauses = _split_into_clauses(page_text)

            for clause_idx, clause in enumerate(raw_clauses):
                refined_chunks = _subdivide_long_clause(clause)
                for sub_idx, chunk_text in enumerate(refined_chunks):
                    if SecurityGuard.detect_prompt_injection(chunk_text):
                        logger.warning(
                            f"Security Alert: Dropped malicious chunk in '{filename}' on page {page_num + 1}."
                        )
                        continue

                    safe_text = SecurityGuard.sanitize_text(chunk_text)
                    all_chunks.append({
                        "content": safe_text,
                        "metadata": {
                            "source": filename,
                            "page": page_num + 1,
                            "clause_id": f"p{page_num + 1}_c{clause_idx + 1}_s{sub_idx + 1}",
                            "char_count": len(safe_text),
                        },
                    })
    finally:
        doc.close()

    return all_chunks