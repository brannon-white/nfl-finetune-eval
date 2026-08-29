"""Parse the (manually downloaded) rulebook PDF into plain text.

Usage:
    python scripts/01_fetch_data.py
"""
from pathlib import Path

from pypdf import PdfReader

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def parse_rulebook() -> None:
    pdf_path = RAW_DIR / "rulebook.pdf"
    if not pdf_path.exists():
        raise SystemExit(
            f"\n{pdf_path} not found.\n"
            "Download the current NFL Official Football Operations rulebook PDF "
            "yourself and place it there, then re-run this script. Not auto-fetched "
            "on purpose -- don't want a scraper against a site's ToS in a public repo."
        )

    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)

    out_path = RAW_DIR / "rulebook.txt"
    out_path.write_text(text)
    print(f"Parsed {len(reader.pages)} pages -> {out_path} ({len(text)} chars)")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    parse_rulebook()
