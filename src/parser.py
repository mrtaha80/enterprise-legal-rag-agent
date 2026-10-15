import re
import pymupdf as fitz
from typing import List, Dict
from src.security import SecurityGuard
from src.logger import logger

# الگوهای جامع‌تر و استاندارد برای شناسایی انواع سرتیترها و مواد قانونی (فارسی و انگلیسی)
CLAUSE_PATTERNS = [
    r"^(?:ماده|بند|تبصره|فصل|بخش|پیوست|صورتجلسه)\s*[\d\u0660-\u0669\u06F0-\u06F9IVXLCDM]+[:\.\-–]",
    r"^(?:Article|Section|Clause|Schedule|Exhibit)\s*[\dIVXLCDM]+[:\.\-–]",
    r"^[\(\（]?\s*[\d\u0660-\u0669\u06F0-\u06F9]+\s*[\)\）\.\-–]",
    r"^[الف ب پ ت ث ج ح خ د ذ ر ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ه ی]\s*[\.\-–\)]"
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
    """خردایش طولانی بر اساس مرز کلمات (Word-Aware Overlap) جهت جلوگیری از نصف شدن کلمات و جملات"""
    if len(clause) <= max_chars:
        return [clause]

    words = clause.split()
    sub_chunks = []
    current_words = []
    current_len = 0

    for word in words:
        current_words.append(word)
        current_len += len(word) + 1

        if current_len >= max_chars:
            sub_chunks.append(" ".join(current_words))
            # اعمال اورلپ بر اساس کلمات (جلوگیری از بریدن کلمه)
            current_words = current_words[-overlap_words:]
            current_len = sum(len(w) + 1 for w in current_words)

    if current_words:
        sub_chunks.append(" ".join(current_words))

    return sub_chunks

def extract_chunks_from_pdf(file_path: str, source_name: str = None) -> List[Dict]:
    """استخراج و تفکیک امن اسناد PDF با پشتیبانی از مرتب‌سازی هندسی بلوک‌ها (جهت مدیریت ستون‌ها و جداول)"""
    doc = fitz.open(file_path)
    filename = source_name if source_name else file_path.split("\\")[-1].split("/")[-1]
    all_chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # استخراج بلوکی همراه با مرتب‌سازی مختصات (sort=True) برای حل مشکل ستون‌بندی و جدول‌ها
        blocks = page.get_text("blocks", sort=True)
        page_text = "\n".join([b[4] for b in blocks if b[4].strip()])
        
        if not page_text:
            continue

        raw_clauses = _split_into_clauses(page_text)

        for clause_idx, clause in enumerate(raw_clauses):
            refined_chunks = _subdivide_long_clause(clause)
            for sub_idx, chunk_text in enumerate(refined_chunks):
                
                # لایه امنیتی صفر-تساهل برای حذف چانک‌های حاوی پرامپت اینجکشن
                if SecurityGuard().detect_prompt_injection(chunk_text):
                    logger.error(f"Security Alert: Dropped malicious chunk in file '{filename}' on page {page_num+1}.")
                    continue

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