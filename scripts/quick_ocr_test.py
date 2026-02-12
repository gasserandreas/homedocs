import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pytesseract
from pdf2image import convert_from_path
from anthropic import Anthropic
import base64

def ocr_tesseract(pdf_path: str) -> str:
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang="deu+fra")
    return text

MAX_API_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB

def ocr_vision_llm(pdf_path: str) -> str | None:
    client = Anthropic()
    images = convert_from_path(pdf_path)

    import io
    content = []
    total_size = 0
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        total_size += buf.tell()
        if total_size > MAX_API_SIZE_BYTES:
            print(f"⚠️  Skipping Vision LLM: images total {total_size / 1024 / 1024:.1f} MB (limit {MAX_API_SIZE_BYTES / 1024 / 1024:.0f} MB)")
            return None
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })

    content.append({
        "type": "text",
        "text": "Extract ALL text from this document. Preserve structure. Return only the extracted text, nothing else.",
    })

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text

def process_pdf(pdf_path: str) -> None:
    pdf = Path(pdf_path)
    print("=" * 60)
    print(f"📄 Testing: {pdf}")
    print("=" * 60)

    print("\n--- Tesseract ---")
    tess_result = ocr_tesseract(str(pdf))
    print(tess_result[:500])

    print("\n--- Vision LLM (Claude) ---")
    llm_result = ocr_vision_llm(str(pdf))
    if llm_result:
        print(llm_result[:500])

    # Speichern zum manuellen Vergleich
    result_dir = Path(f"test_documents/results/{pdf.stem}")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "tesseract.txt").write_text(tess_result)
    if llm_result:
        (result_dir / "vision_llm.txt").write_text(llm_result)
    print(f"\n✅ Results saved to {result_dir}/")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_ocr_test.py <folder_path>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory")
        sys.exit(1)

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in '{folder}'")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s) in '{folder}'\n")
    for pdf in pdfs:
        process_pdf(str(pdf))
