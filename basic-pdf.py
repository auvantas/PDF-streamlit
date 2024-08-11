from typing import List, Dict, Optional, Literal, Tuple
from PIL import Image
import streamlit as st
import os
import tempfile
import json
import time
import requests
import openai
import copy

from surya.languages import CODE_TO_LANGUAGE, LANGUAGE_TO_CODE
# Removed direct import from surya
from marker.ocr.tesseract import LANGUAGE_TO_TESSERACT_CODE, TESSERACT_CODE_TO_LANGUAGE
from marker.settings import settings
from marker.pdf.utils import replace_langs_with_codes, validate_langs, langs_to_ids
from marker.pdf.extract_text import get_text_blocks
from marker.ocr.recognition import run_ocr
from marker.layout.layout import annotate_block_types
from marker.layout.order import sort_blocks_in_reading_order
from marker.equations.equations import replace_equations
from marker.tables.table import format_tables
from marker.cleaners.headers import filter_header_footer, filter_common_titles
from marker.cleaners.code import identify_code_blocks, indent_blocks
from marker.cleaners.bullets import replace_bullets
from marker.cleaners.headings import split_heading_blocks
from marker.cleaners.fontstyle import find_bold_italic
from marker.cleaners.text import cleanup_text
from marker.postprocessors.editor import edit_full_text
from marker.postprocessors.markdown import merge_spans, merge_lines, get_full_text
from marker.images.extract import extract_images
from marker.images.save import images_to_dict
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from marker.output import markdown_exists, save_markdown
from marker.title_page import generate_title_page
from marker.toc import generate_table_of_contents
from marker.citations import extract_citations, format_citations
from marker.document_structure import analyze_structure, apply_template
from marker.content_enhancement import summarize_section, improve_flow
from marker.visual_integration import insert_visuals
from marker.export import export_document

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

st.sidebar.header("Advanced Settings")
OCR_ALL_PAGES = st.sidebar.checkbox("Force OCR on all pages", value=False, help="Useful if table layouts aren't recognized properly or if there is garbled text.")
OCR_ENGINE = st.sidebar.selectbox("OCR Engine", ["surya", "ocrmypdf"], help="Select the OCR engine to use.")
PAGINATE_OUTPUT = st.sidebar.checkbox("Paginate Output", value=False, help="Add a horizontal rule between pages.")
EXTRACT_IMAGES = st.sidebar.checkbox("Extract Images", value=True, help="Extract images and save separately.")
MIN_LENGTH = st.sidebar.number_input("Minimum PDF Length (characters)", value=10000, help="Minimum number of characters for a PDF to be processed.")

# --- Model Data Processing Parameters ---
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_REFERENCE_MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
]
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2048

# Model-specific rate limiting parameters (tokens per minute)
MODEL_RATE_LIMITS = {
    "llama3-70b-8192": 6000,
    "llama3-8b-8192": 6000,
    "mixtral-8x7b-32768": 5000,
    "gemma-7b-it": 6000,
}

DELAY_BETWEEN_CALLS = 20  # seconds

# Retry parameters
MAX_RETRIES = 5
BASE_WAIT = 1  # seconds
MAX_WAIT = 60  # seconds

# --- Session State Initialization ---
if "job_running" not in st.session_state:
    st.session_state.job_running = False

# --- Constants and Setup ---
DEBUG = int(os.environ.get("DEBUG", "0"))

# Load API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]
GROQ_API_KEY = st.secrets["groq"]["GROQ_API_KEY"]

LANGUAGE_TO_TESSERACT_CODE = {
    "Afrikaans": "afr", "Amharic": "amh", "Arabic": "ara", "Assamese": "asm",
    "Azerbaijani": "aze", "Belarusian": "bel", "Bulgarian": "bul", "Bengali": "ben",
    "Tibetan": "bod", "Bosnian": "bos", "Catalan": "cat", "Cebuano": "ceb",
    "Czech": "ces", "Welsh": "cym", "Danish": "dan", "German": "deu",
    "Greek": "ell", "English": "eng", "Spanish": "spa", "Estonian": "est",
    "Basque": "eus", "Persian": "fas", "Finnish": "fin", "French": "fra",
    "Irish": "gle", "Galician": "glg", "Gujarati": "guj", "Hebrew": "heb",
    "Hindi": "hin", "Croatian": "hrv", "Hungarian": "hun", "Indonesian": "ind",
    "Icelandic": "isl", "Italian": "ita", "Japanese": "jpn", "Georgian": "kat",
    "Kazakh": "kaz", "Khmer": "khm", "Kannada": "kan", "Korean": "kor",
    "Kurdish": "kur", "Lao": "lao", "Lithuanian": "lit", "Latvian": "lav",
    "Malayalam": "mal", "Marathi": "mar", "Macedonian": "mkd", "Mongolian": "mon",
    "Malay": "msa", "Maltese": "mlt", "Burmese": "mya", "Nepali": "nep",
    "Dutch": "nld", "Norwegian": "nor", "Oriya": "ori", "Punjabi": "pan",
    "Polish": "pol", "Portuguese": "por", "Romanian": "ron", "Russian": "rus",
    "Sinhala": "sin", "Slovak": "slk", "Slovenian": "slv", "Albanian": "sqi",
    "Serbian": "srp", "Swedish": "swe", "Swahili": "swa", "Tamil": "tam",
    "Telugu": "tel", "Thai": "tha", "Tagalog": "tgl", "Turkish": "tur",
    "Ukrainian": "ukr", "Urdu": "urd", "Uzbek": "uzb", "Vietnamese": "vie",
    "Chinese (Simplified)": "chi_sim", "Chinese (Traditional)": "chi_tra"
}
TESSERACT_CODE_TO_LANGUAGE = {v: k for k, v in LANGUAGE_TO_TESSERACT_CODE.items()}

# --- Rate Limiting Helper ---
def rate_limit(model):
    """Delays the execution to respect the rate limit of the model."""
    limit = MODEL_RATE_LIMITS.get(model)
    if limit:
        time.sleep(60 / limit)

# --- Retry Helper ---
def retry_with_backoff(func, *args, **kwargs):
    """Retries a function call with exponential backoff."""
    num_retries = 0
    wait_time = BASE_WAIT
    while num_retries < MAX_RETRIES:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            num_retries += 1
            wait_time = min(wait_time * 2, MAX_WAIT)
    st.error(f"Failed after {MAX_RETRIES} retries. Giving up.")
    return None

# --- Enhanced PDF Conversion Function ---
def enhanced_convert_single_pdf(
    fname: str,
    model_lst: List,
    max_pages: int = None,
    start_page: int = None,
    metadata: Optional[Dict] = None,
    langs: Optional[List[str]] = None,
    document_template: str = "Default",
    citation_style: str = "APA",
    ocr_all_pages: bool = False,
    ocr_engine: str = "surya",
    paginate_output: bool = False,
    extract_images: bool = True,
    min_length: int = 10000
) -> Tuple[str, Dict[str, Image.Image], Dict]:
    # Update settings
    settings.OCR_ALL_PAGES = ocr_all_pages
    settings.OCR_ENGINE = ocr_engine
    settings.PAGINATE_OUTPUT = paginate_output
    settings.EXTRACT_IMAGES = extract_images

    # Existing conversion logic
    full_text, images, out_metadata = convert_single_pdf(
        fname, model_lst, max_pages, start_page, metadata, langs
    )
    
    # Check minimum length
    if len(full_text) < min_length:
        raise ValueError(f"PDF content is too short ({len(full_text)} characters). Minimum length is {min_length}.")
# --- MoA Functions ---
def generate_together(
    model,
    messages,
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    streaming=False,
):
    rate_limit(model)
    output = retry_with_backoff(
        requests.post,
        "https://api.groq.com/openai/v1/chat/completions",
        json={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature if temperature > 1e-4 else 0,
            "messages": messages,
        },
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
    )
    if output is None:
        return None
    if "error" in output.json():
        st.error(output.json())
        if output.json()["error"]["type"] == "invalid_request_error":
            st.write("Input + output is longer than max_position_id.")
            return None
    return output.json()["choices"][0]["message"]["content"].strip()

def generate_openai(model, messages, max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE):
    rate_limit(model)  # Apply rate limiting
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output = completion.choices[0].message.content
    return output.strip()

def inject_references_to_messages(messages, references):
    messages = copy.deepcopy(messages)
    system = f"""You have been provided with a set of responses from various models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""
    for i, reference in enumerate(references):
        system += f"\n{i+1}. {reference}"
    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system
    else:
        messages = [{"role": "system", "content": system}] + messages
    return messages

def generate_with_references(
    model=DEFAULT_MODEL,
    messages=[],
    references=[],
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
):
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)
    return generate_together(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

def langs_to_ids(langs: List[str]):
    unique_langs = list(set(langs))
    _, lang_tokens = lang_tokenize("", unique_langs)
    return lang_tokens

def replace_langs_with_codes(langs):
    if settings.OCR_ENGINE == "surya":
        for i, lang in enumerate(langs):
            if lang.title() in LANGUAGE_TO_CODE:
                langs[i] = LANGUAGE_TO_CODE[lang.title()]
    else:
        for i, lang in enumerate(langs):
            if lang in LANGUAGE_TO_CODE:
                langs[i] = LANGUAGE_TO_TESSERACT_CODE[lang]
    return langs

def validate_langs(langs):
    if settings.OCR_ENGINE == "surya":
        for lang in langs:
            if lang not in CODE_TO_LANGUAGE:
                raise ValueError(f"Invalid language code {lang} for Surya OCR")
    else:
        for lang in langs:
            if lang not in TESSERACT_CODE_TO_LANGUAGE:
                raise ValueError(f"Invalid language code {lang} for Tesseract")

# --- Streamlit App ---
st.title("Enhanced PDF to Markdown Converter & Report Generator")

# --- Sidebar ---
st.sidebar.header("Options")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files (one at a time)", type=["pdf"], accept_multiple_files=False
)

# Language selection for OCR
selected_languages = st.sidebar.multiselect(
    "Select Languages for OCR",
    list(LANGUAGE_TO_TESSERACT_CODE.keys()),
    ["English"],  # Default language
)

model_name = st.sidebar.selectbox(
    "Select Model",
    DEFAULT_REFERENCE_MODELS,
    help="Select a model for report generation.",
)

output_format = st.sidebar.radio(
    "Select Output Format",
    ("Markdown", "Copyable Text"),
    help="Choose how you want to view the output.",
)

document_template = st.sidebar.selectbox(
    "Document Template",
    ["Default", "Research Paper", "Book Chapter", "Technical Report"],
    help="Select a template for your document structure."
)

citation_style = st.sidebar.selectbox(
    "Citation Style",
    ["APA", "MLA", "Chicago"],
    help="Choose the citation style for references."
)

# --- Main Content Area ---
if not st.session_state.job_running and uploaded_files:
    st.session_state.job_running = True
    try:
        model_lst = load_all_models(langs=selected_languages)
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, uploaded_files.name), "wb") as f:
                f.write(uploaded_files.getbuffer())
            with st.spinner(f"Converting and enhancing {uploaded_files.name}..."):
                final_document, images, out_metadata = enhanced_convert_single_pdf(
                    os.path.join(tmpdir, uploaded_files.name),
                    model_lst,
                    langs=selected_languages,
                    document_template=document_template,
                    citation_style=citation_style,
                    ocr_all_pages=OCR_ALL_PAGES,
                    ocr_engine=OCR_ENGINE,
                    paginate_output=PAGINATE_OUTPUT,
                    extract_images=EXTRACT_IMAGES,
                    min_length=MIN_LENGTH
                )

            # Display enhanced document
            st.markdown(f"## Enhanced {uploaded_files.name}")
            if output_format == "Markdown":
                st.markdown(final_document)
            elif output_format == "Copyable Text":
                st.code(final_document)

            # Download options
            st.download_button(
                label="Download Markdown",
                data=final_document,
                file_name="converted_document.md",
                mime="text/markdown",
            )

            # Display extracted images
            st.markdown("### Extracted Images")
            for filename, image in images.items():
                st.image(image, caption=filename)

            # Export options
            st.header("Export Options")
            export_format = st.selectbox("Export Format", ["PDF", "DOCX", "HTML"])
            if st.button("Export Document"):
                exported_file = export_document(final_document, export_format)
                st.download_button(
                    label=f"Download {export_format}",
                    data=exported_file,
                    file_name=f"enhanced_document.{export_format.lower()}",
                    mime=f"application/{export_format.lower()}"
                )

            # Content Enhancement Tools
            st.header("Content Enhancement")
            if st.button("Summarize Document"):
                summary = summarize_section(final_document)
                st.markdown("### Document Summary")
                st.markdown(summary)

            # Visual Integration
            st.header("Visual Integration")
            uploaded_visual = st.file_uploader("Upload a chart or graph", type=["png", "jpg", "jpeg"])
            if uploaded_visual:
                insert_position = st.text_input("Enter the heading where you want to insert the visual:")
                if st.button("Insert Visual"):
                    final_document = insert_visuals(final_document, uploaded_visual, insert_position)
                    st.markdown("Visual inserted successfully. You can now export the updated document.")

            # Report Generation
            st.header("Report Generation")
            report_topic = st.text_input("Enter the topic for your report:")
            if st.button("Generate Report"):
                if report_topic:
                    with st.spinner("Generating report..."):
                        reference_outputs = []
                        for ref_model in DEFAULT_REFERENCE_MODELS:
                            messages = [
                                {
                                    "role": "user",
                                    "content": f"Gather information about {report_topic} from the provided documents.",
                                },
                            ]
                            ref_output = generate_with_references(
                                model=ref_model,
                                messages=messages,
                                references=[final_document],
                            )
                            if ref_output:
                                reference_outputs.append(ref_output)
                            time.sleep(DELAY_BETWEEN_CALLS)  # Delay between calls to reference models

                        # Generate final report with default model
                        messages = [
                            {
                                "role": "user",
                                "content": f"Create a structured report about {report_topic}, using the information gathered. Ensure to include citations and references.",
                            },
                        ]
                        report = generate_with_references(
                            model=DEFAULT_MODEL, messages=messages, references=reference_outputs
                        )
                    st.markdown("### Generated Report")
                    st.markdown(report)

                    # Report Modification
                    st.header("Report Modification")
                    modification_request = st.text_input(
                        "Enter your modification request (e.g., 'expand on XYZ topic'):"
                    )
                    if st.button("Modify Report"):
                        if modification_request:
                            with st.spinner("Modifying report..."):
                                messages = [
                                    {
                                        "role": "user",
                                        "content": f"Here's the current report:\n\n{report}\n\n{modification_request}",
                                    }
                                ]
                                modified_report = generate_with_references(
                                    model=DEFAULT_MODEL,
                                    messages=messages,
                                    references=reference_outputs,
                                )
                            st.markdown("### Modified Report")
                            st.markdown(modified_report)
                else:
                    st.warning("Please enter a report topic.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        st.session_state.job_running = False  # Allow new jobs after finishing
elif st.session_state.job_running:
    st.warning("A job is currently running. Please wait for it to finish.")
else:
    st.warning("Please upload a PDF file.")
