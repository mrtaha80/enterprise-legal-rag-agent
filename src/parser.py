import re
import pymupdf as fitz
from typing import List, Dict
from src.security import SecurityGuard
from src.logger import logger

CLAUSE_PATTERNS = [
    r"^(?:ماده|بند|تبصره|فصل|بخش)\s*[\d\u0660-\u0669\u06F0-\u06F9IVXLCDM]+[:\.\-–]",
    r"^(?:Article|Section|Clause)\s*[\dIVXLCDM]+[:\.\-–]",
    r"^\d+[\.\-–]\s+",
]

COMBINED_PATTERN = re.compile("|".join(CLAUSE_PATTERNS), re.IGNORECASE | re.MULTILINE)

def _split_into_clauses(text: str) -> List[str]:
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

def _subdivide_long_clause(clause: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    if len(clause) <= max_chars:
        return [clause]

    paragraphs = clause.split("\n")
    sub_chunks = []
    current_chunk = ""

    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        if len(current_chunk) + len(p_clean) + 1 <= max_chars:
            current_chunk = f"{current_chunk}\n{p_clean}".strip()
        else:
            if current_chunk:
                sub_chunks.append(current_chunk)
                current_chunk = current_chunk[-overlap:] + "\n" + p_clean
            else:
                for idx in range(0, len(p_clean), max_chars - overlap):
                    sub_chunks.append(p_clean[idx:idx + max_chars])
                current_chunk = ""

    if current_chunk:
        sub_chunks.append(current_chunk)

    return sub_chunks

def extract_chunks_from_pdf(file_path: str, source_name: str = None) -> List[Dict]:
    doc = fitz.open(file_path)
    filename = source_name if source_name else file_path.split("\\")[-1].split("/")[-1]
    all_chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text").strip()
        if not page_text:
            continue

        # دفاع امنیتی: بررسی تزریق پرامپت در سطح صفحه PDF
        if SecurityGuard.detect_prompt_injection(page_text):
            logger.warning(f"Potential Prompt Injection detected in file '{filename}' on page {page_num+1}!")
            # می‌توانیم متن‌های مشکوک را خنثی یا برچسب‌گذاری کنیم
            page_text = "[SECURITY WARNING: Potential malicious prompt injection neutralized in source]\n" + page_text

        raw_clauses = _split_into_clauses(page_text)

        for clause_idx, clause in enumerate(raw_clauses):
            refined_chunks = _subdivide_long_clause(clause)
            for sub_idx, chunk_text in enumerate(refined_chunks):
                all_chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1,
                        "clause_id": f"p{page_num+1}_c{clause_idx+1}_s{sub_idx+1}",
                        "char_count": len(chunk_text)
                    }
                })

    doc.close()
    return all_chunks