import fitz
import re

def clean_persian_text(text: str) -> str:
    text = re.sub(r'[\r\f\v]', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_text_with_overlap(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]

def extract_chunks_from_pdf(pdf_path: str, chunk_size=500, chunk_overlap=80) -> list[dict]:
    doc = fitz.open(pdf_path)
    final_chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text")
        cleaned_text = clean_persian_text(raw_text)
        
        if not cleaned_text:
            continue

        page_splits = split_text_with_overlap(cleaned_text, chunk_size, chunk_overlap)
        for chunk in page_splits:
            final_chunks.append({
                "content": chunk,
                "metadata": {
                    "source": pdf_path.split("/")[-1],
                    "page": page_num + 1
                }
            })
    return final_chunks
