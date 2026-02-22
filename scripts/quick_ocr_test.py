"""
OCR Comparison Script
=====================
Runs two OCR methods on every PDF in a given folder and saves the results for
manual comparison. Already-processed files are skipped automatically.

Usage (run from the project root):
    python scripts/quick_ocr_test.py <folder_with_pdfs>

Example:
    python scripts/quick_ocr_test.py test_documents/input

Output:
    Results are written to test_documents/results/<pdf-stem>/
        tesseract.txt   -- text extracted by Tesseract (local, offline)
        vision_llm.txt  -- text extracted by Claude Vision (requires ANTHROPIC_API_KEY)

    A file is considered already processed when its result folder exists.
    Re-run the script any time; only new PDFs will be processed.

Requirements:
    pip install pytesseract pdf2image anthropic python-dotenv
    Tesseract must be installed on the system (brew install tesseract on macOS).
    ANTHROPIC_API_KEY must be set in a .env file or the environment.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import io

import pytesseract
from pdf2image import convert_from_path
from anthropic import Anthropic
from anthropic.types import TextBlock
import base64


def ocr_tesseract(pdf_path: str) -> str:
    """Extract text from a PDF using Tesseract, one page at a time.

    Each page is processed individually and results are joined with page markers.
    """
    images = convert_from_path(pdf_path)
    page_texts = []
    for img in images:
        page_texts.append(pytesseract.image_to_string(img, lang="deu+fra"))

    if len(page_texts) == 1:
        return page_texts[0]
    return "\n\n".join(f"--- Page {i} ---\n{text}" for i, text in enumerate(page_texts, start=1))


# Anthropic API enforces a 5 MB limit on the base64-encoded image payload.
# Base64 adds ~33 % overhead, so the raw PNG must stay below 5 MB × (3/4) = 3.75 MB.
MAX_PAGE_SIZE_BYTES = 5 * 1024 * 1024 * 3 // 4  # ~3.75 MB


def _encode_page(img) -> bytes:
    """Return the PNG bytes for an image, scaling it down if needed to stay under the API limit.

    Repeatedly reduces the image to 75 % of its size until it fits.
    Raises RuntimeError if the image cannot be shrunk enough.
    """
    scale = 1.0
    current = img
    while True:
        buf = io.BytesIO()
        current.save(buf, format="PNG")
        data = buf.getvalue()
        if len(data) <= MAX_PAGE_SIZE_BYTES:
            return data
        scale *= 0.75
        if scale < 0.1:
            raise RuntimeError(f"Image could not be reduced below {MAX_PAGE_SIZE_BYTES / 1024 / 1024:.0f} MB")
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        current = img.resize((new_w, new_h))
        print(f"   ↳ Resized to {scale:.0%} ({new_w}×{new_h}) to fit API limit")


def ocr_vision_llm(pdf_path: str) -> str | None:
    """Extract text from a PDF by sending each page as a separate API request to Claude.

    Pages that are too large are scaled down automatically before sending.
    All page results are combined into a single string. Returns None if no
    pages could be processed.
    """
    client = Anthropic()
    images = convert_from_path(pdf_path)

    page_texts = []
    for i, img in enumerate(images, start=1):
        try:
            png_data = _encode_page(img)
        except RuntimeError as e:
            print(f"⚠️  Page {i}: skipping — {e}")
            page_texts.append(f"[Page {i}: skipped — could not resize to fit API limit]")
            continue

        print(f"   Page {i}/{len(images)} ({len(png_data) / 1024:.0f} KB) → Claude...")
        b64 = base64.standard_b64encode(png_data).decode()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": "Extract ALL text from this page. Preserve structure. Return only the extracted text, nothing else."},
            ]}],
        )
        text_block = next(b for b in response.content if isinstance(b, TextBlock))
        page_texts.append(text_block.text)

    if not page_texts:
        return None

    if len(page_texts) == 1:
        return page_texts[0]
    return "\n\n".join(f"--- Page {i} ---\n{text}" for i, text in enumerate(page_texts, start=1))


def process_pdf(pdf_path: str) -> None:
    """Run both OCR methods on a single PDF and save results to disk."""
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

    # Save both outputs under test_documents/results/<stem>/ for manual review.
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

    # A PDF is considered processed when its result folder already exists.
    results_base = Path("test_documents/results")
    pending = [pdf for pdf in pdfs if not (results_base / pdf.stem).is_dir()]

    print(f"Found {len(pdfs)} PDF(s) in '{folder}', {len(pdfs) - len(pending)} already processed, {len(pending)} pending\n")
    if not pending:
        print("Nothing to do.")
        sys.exit(0)

    for pdf in pending:
        process_pdf(str(pdf))
