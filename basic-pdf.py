import streamlit as st
import os
import tempfile
import json
import time
import requests
import openai
import copy
from typing import List, Dict, Optional, Literal, Tuple
from PIL import Image

import psutil
import pypdfium2 as pdfium
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from marker.output import markdown_exists, save_markdown
from marker.settings import settings
from datasets.utils.logging import disable_progress_bar
from marker.pdf.utils import (
    replace_langs_with_codes,
    validate_langs,
    find_filetype,
    sort_block_group,
)
from marker.pdf.extract_text import get_text_blocks
from marker.ocr.detection import surya_detection
from marker.ocr.recognition import run_ocr
from marker.layout.layout import surya_layout, annotate_block_types
from marker.layout.order import surya_order, sort_blocks_in_reading_order
from marker.equations.equations import replace_equations
from marker.tables.table import format_tables
from marker.cleaners.headers import filter_header_footer, filter_common_titles
from marker.cleaners.code import identify_code_blocks, indent_blocks
from marker.cleaners.bullets import replace_bullets
from marker.cleaners.headings import split_heading_blocks
from marker.cleaners.fontstyle import find_bold_italic
from marker.postprocessors.editor import edit_full_text
from marker.postprocessors.markdown import (
    merge_spans,
    merge_lines,
    get_full_text,
)
from marker.cleaners.text import cleanup_text
from marker.images.extract import extract_images
from marker.images.save import images_to_dict
from marker.debug.data import dump_bbox_debug_data, dump_equation_debug_data
from marker.equations.inference import get_total_texify_tokens, get_latex_batched
from marker.pdf.images import render_bbox_image

# New imports for enhanced features
from marker.title_page import generate_title_page
from marker.toc import generate_table_of_contents
from marker.citations import extract_citations, format_citations
from marker.document_structure import analyze_structure, apply_template
from marker.content_enhancement import summarize_section, improve_flow
from marker.visual_integration import insert_visuals
from marker.export import export_document

disable_progress_bar()

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

# --- RAM Management ---
MAX_RAM_USAGE_PERCENT = 80  # Set maximum RAM usage threshold

def check_ram_usage():
    """Checks if RAM usage is below the threshold."""
    ram_percent = psutil.virtual_memory().percent
    return ram_percent < MAX_RAM_USAGE_PERCENT

# --- Session State Initialization ---
if "job_running" not in st.session_state:
    st.session_state.job_running = False

# --- Constants and Setup ---
DEBUG = int(os.environ.get("DEBUG", "0"))

# Load API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]
GROQ_API_KEY = st.secrets["groq"]["GROQ_API_KEY"]

LANGUAGE_TO_TESSERACT_CODE = {
    # ... (keep the existing language codes)
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
    batch_multiplier: int = 1,
    document_template: str = "Default",
    citation_style: str = "APA"
) -> Tuple[str, Dict[str, Image.Image], Dict]:
    # Existing conversion logic
    full_text, images, out_metadata = convert_single_pdf(
        fname, model_lst, max_pages, start_page, metadata, langs, batch_multiplier
    )
    
    # Extract metadata for title page
    metadata = extract_metadata(fname)
    title_page = generate_title_page(metadata)
    
    # Generate table of contents
    toc = generate_table_of_contents(full_text)
    
    # Extract and format citations
    citations = extract_citations(full_text)
    formatted_citations = format_citations(citations, style=citation_style)
    
    # Analyze and structure the document
    structured_text = analyze_structure(full_text)
    
    # Apply document template
    templated_text = apply_template(structured_text, template=document_template)
    
    # Enhance content
    enhanced_text = improve_flow(templated_text)
    
    # Combine all elements
    final_document = f"{title_page}\n\n{toc}\n\n{enhanced_text}\n\n{formatted_citations}"
    
    return final_document, images, out_metadata

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

def generate_together_stream(
    model,
    messages,
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
):
    rate_limit(model)  # Apply rate limiting
    endpoint = "https://api.groq.com/openai/v1/"
    client = openai.OpenAI(api_key=GROQ_API_KEY, base_url=endpoint)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        stream=True,
    )
    return response

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
    system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

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
    generate_fn=generate_together,
):
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)
    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

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
    st.session_state.job_running = True  # Prevent new jobs while one is running
    if check_ram_usage():
        try:
            # Load models
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
                        citation_style=citation_style
                    )

                # Display enhanced document
                st.markdown(f"## Enhanced {uploaded_files.name}")
                if output_format == "Markdown":
                    st.markdown(final_document)
                elif output_format == "Copyable Text":
                    st.code(final_document)

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
                                    generate_fn=generate_together
                                    if "groq" in ref_model
                                    else generate_openai,
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
    else:
        st.warning(
            f"RAM usage is too high ({psutil.virtual_memory().percent}%). "
            "Please wait for the current job to finish or try uploading a smaller PDF."
        )
elif st.session_state.job_running:
    st.warning("A job is currently running. Please wait for it to finish.")
else:
    st.warning("Please upload a PDF file.")

# Clean up CUDA Memory
from marker.utils import flush_cuda_memory
flush_cuda_memory()
