"""
Utility functions for extracting and analysing VPAT documents using either
direct text extraction or the OpenAI GPT‑4o vision model.  The goal of
this module is to produce a predictable, line‑based summary of all
WCAG and Section 508 success criteria contained in a VPAT.  Each
success criterion is represented on its own line using the format:

    <criterion> <description> - <support level> - <remarks>

This structured output makes it easy for the front‑end parser to
convert AI results into editable tables.  When the PDF contains
selectable text (as most VPATs do), the analyser extracts the text
directly and feeds it to GPT‑4o with a strict prompt.  If the PDF is
scanned or yields very little text, the analyser falls back to
rendering each page as an image and sending those images to GPT‑4o
vision with a similar prompt.

Note: This module expects an environment variable `OPENAI_API_KEY` to
be set for authenticating with the OpenAI API.  If the key is not
present, the `OpenAI` client will raise an error when invoked.
"""

import base64
import os
from typing import List

import fitz  # PyMuPDF for PDF rendering and text extraction
from openai import OpenAI

# Initialise a single OpenAI client.  The API key is read from the
# OPENAI_API_KEY environment variable.  If you need to customise the
# client (e.g. for proxies), you can adjust this accordingly.
client = OpenAI()


def extract_vpat_text(pdf_path: str) -> str:
    """Extract plain text from a PDF using PyMuPDF.

    Args:
        pdf_path: Path to the VPAT PDF file.

    Returns:
        The concatenated text of all pages.  If the PDF contains no
        embedded text (for example, if it is scanned), this may return
        an empty string or a very short string.
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # get_text("text") extracts selectable text.  Using
            # get_text("text") instead of get_text("blocks") keeps the
            # order closer to reading order.
            text += page.get_text("text") + "\n"
    return text


def analyse_text_with_chat(text: str) -> str:
    """Analyse extracted VPAT text using GPT‑4o and return a line‑based summary.

    The prompt instructs GPT‑4o to output each success criterion on a
    separate line with a defined format.  This makes the model’s
    response much easier to parse on the client side.

    Args:
        text: The raw VPAT text extracted from the PDF.

    Returns:
        A string containing zero or more lines in the format
        `<criterion> <description> - <support level> - <remarks>`.
    """
    prompt = (
        "You're a VPAT accessibility parser. Read the following text and, for each "
        "WCAG 2.x or Section 508 success criterion, output a single line in this exact format:\n"
        "<criterion> <description> - <support level> - <remarks>\n\n"
        "Only include success criteria. Do not include headings or explanatory text. "
        "The support level must be one of: Supports, Partially Supports, Does Not Support, or Not Applicable. "
        "The remarks should be a brief summary of the evaluator's comments.\n\n"
        f"{text}"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert VPAT and accessibility evaluator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def analyse_page_image(image_path: str) -> str:
    """Analyse a single page image using GPT‑4o vision and return a summary.

    This function encodes the image in base64 and sends it to GPT‑4o
    along with a structured prompt.  The model is asked to produce one
    line per success criterion in the same format as the text analyser.

    Args:
        image_path: Path to the PNG image of a PDF page.

    Returns:
        A string containing zero or more lines summarising the success
        criteria found on that page.  If no criteria are found, returns
        an empty string.
    """
    with open(image_path, "rb") as f:
        image_data = f.read()
    b64_image = base64.b64encode(image_data).decode("utf-8")
    prompt = (
        "You're a VPAT accessibility parser. For each WCAG 2.x or Section 508 "
        "success criterion found on this page, output a single line in the "
        "format: <criterion> <description> - <support level> - <remarks>. "
        "Only include success criteria; skip other text."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert VPAT and accessibility evaluator."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                ],
            },
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def analyse_scanned_pdf(pdf_path: str) -> str:
    """Fallback analysis for scanned PDFs.

    If a PDF has little or no embedded text, we consider it scanned.  This
    function rasterises each page at 200 DPI, sends the image to GPT‑4o
    vision with a structured prompt, and concatenates the per‑page
    results.  Empty lines are filtered out.

    Args:
        pdf_path: Path to the scanned PDF.

    Returns:
        A newline‑separated string of success criteria lines.  May be
        empty if no criteria are detected.
    """
    lines: List[str] = []
    with fitz.open(pdf_path) as doc:
        os.makedirs("images", exist_ok=True)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            image_path = f"images/page_{i+1}.png"
            pix.save(image_path)
            result = analyse_page_image(image_path)
            if result:
                # Split by newlines to avoid concatenating multiple lines into one
                lines.extend([line.strip() for line in result.split("\n") if line.strip()])
    return "\n".join(lines)


def analyze_pdf(pdf_path: str) -> str:
    """Analyse a VPAT PDF and return a structured, line‑based summary.

    The high‑level logic is:

    1. Extract text using `extract_vpat_text()`.  If the extracted text
       contains more than ~50 characters, assume the PDF is text‑based
       and analyse it with GPT‑4o via the chat endpoint.
    2. If the extracted text is very short (indicating a scanned PDF),
       fall back to rendering each page as an image and analysing it
       with GPT‑4o vision.
    3. Return the model’s output directly as a string.  This string
       should contain one success criterion per line.

    Args:
        pdf_path: Path to the VPAT PDF file.

    Returns:
        A plain‑text summary of the success criteria in the VPAT.  If
        neither text extraction nor vision analysis yields any
        criteria, an empty string is returned.
    """
    # First try to get text from the PDF.  Many modern PDFs embed text
    # directly; in those cases we want to avoid using the more
    # expensive and less reliable vision API.
    text = extract_vpat_text(pdf_path)
    if text and len(text.strip()) > 50:
        try:
            return analyse_text_with_chat(text)
        except Exception as e:
            # If the chat call fails for any reason, fall back to vision
            print(f"Chat analysis failed: {e}. Falling back to vision.")
    # If no text or chat analysis failed, use the vision model on images
    return analyse_scanned_pdf(pdf_path)
