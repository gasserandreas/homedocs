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

def ocr_vision_llm(pdf_path: str) -> str:
    client = Anthropic()
    images = convert_from_path(pdf_path)

    content = []
    for img in images:
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
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

if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "test_documents/invoices/invoice_01.pdf"

    print("=" * 60)
    print(f"📄 Testing: {pdf}")
    print("=" * 60)

    print("\n--- Tesseract ---")
    tess_result = ocr_tesseract(pdf)
    print(tess_result[:500])

    print("\n--- Vision LLM (Claude) ---")
    llm_result = ocr_vision_llm(pdf)
    print(llm_result[:500])

    # Speichern zum manuellen Vergleich
    stem = Path(pdf).stem
    Path(f"test_documents/results/{stem}_tesseract.txt").write_text(tess_result)
    Path(f"test_documents/results/{stem}_vision_llm.txt").write_text(llm_result)
    print(f"\n✅ Results saved to test_documents/results/")
