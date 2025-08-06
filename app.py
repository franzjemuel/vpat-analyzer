from flask import Flask, request, jsonify, render_template
from analyze_vpat import analyze_pdf
import os
from openai import OpenAI

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze an uploaded PDF VPAT file using the AI vision model.

    This endpoint expects a PDF file under the 'file' form field.  The file is saved
    to the uploads directory, passed to analyze_pdf() (which calls GPT‑4o vision), and
    then returns the raw AI output as plain text.  Returning plain text rather than
    a JSON object lets the front‑end parse the result directly or fall back to its
    own local parser when necessary.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    # Run the AI vision model on the PDF.  This should return a string
    # containing either structured JSON or human‑readable lines such as
    # "1.1.1 Non‑text Content - Supports - Images have alt text".
    result = analyze_pdf(filepath)
    # If analyze_pdf returns a dict or other object, convert it to a string
    if not isinstance(result, str):
        result = str(result)
    return result, 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    """
    Analyze raw VPAT text using a chat model.

    The request body should contain a JSON object with a 'text' field.  This endpoint
    sends the text to OpenAI and asks it to extract a list of success criteria in a
    simple, line‑based format.  The AI’s response is returned verbatim as plain text.
    """
    content = request.get_json()
    if not content or 'text' not in content:
        return jsonify({'error': 'No text provided'}), 400
    text = content['text']
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Explicitly instruct the AI to produce a predictable, line‑based format.
    prompt = (
        "You're a VPAT accessibility parser. Read this raw VPAT text and extract a list of WCAG 2.x "
        "and Section 508 success criteria. For each criterion, return:\n"
        "1. The success criterion number (e.g., 1.1.1)\n"
        "2. A short description (e.g., Non-text Content)\n"
        "3. The support level (Supports, Partially Supports, Does Not Support, or Not Applicable)\n"
        "4. A brief evaluator remark.\n\n"
        "Output each result on a separate line in the format:\n"
        "<criterion> <description> - <support level> - <remark>\n\n"
        f"{text}"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert VPAT and accessibility evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  # lower temperature for more deterministic output
    )
    parsed_text = response.choices[0].message.content
    # Return the AI’s output as plain text
    return parsed_text, 200, {'Content-Type': 'text/plain; charset=utf-8'}

if __name__ == '__main__':
    app.run(debug=True)
