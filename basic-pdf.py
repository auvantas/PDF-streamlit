import streamlit as st
import os
import tempfile
import json
import time
import requests
import openai
import copy
from typing import List, Dict, Optional, Literal
from PIL import Image

import pypdfium2 as pdfium  # Needs to be imported early to avoid warnings
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from marker.output import markdown_exists, save_markdown
from marker.utils import flush_cuda_memory
from marker.settings import settings
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

# --- Constants and Setup ---
DEBUG = int(os.environ.get("DEBUG", "0"))

# Load API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]
GROQ_API_KEY = st.secrets["groq"]["GROQ_API_KEY"]

default_reference_models = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

LANGUAGE_TO_TESSERACT_CODE = {
    "Afrikaans": "afr",
    "Amharic": "amh",
    "Arabic": "ara",
    "Assamese": "asm",
    "Azerbaijani": "aze",
    "Belarusian": "bel",
    "Bulgarian": "bul",
    "Bengali": "ben",
    "Breton": "bre",
    "Bosnian": "bos",
    "Catalan": "cat",
    "Czech": "ces",
    "Welsh": "cym",
    "Danish": "dan",
    "German": "deu",
    "Greek": "ell",
    "English": "eng",
    "Esperanto": "epo",
    "Spanish": "spa",
    "Estonian": "est",
    "Basque": "eus",
    "Persian": "fas",
    "Finnish": "fin",
    "French": "fra",
    "Western Frisian": "fry",
    "Irish": "gle",
    "Scottish Gaelic": "gla",
    "Galician": "glg",
    "Gujarati": "guj",
    "Hausa": "hau",
    "Hebrew": "heb",
    "Hindi": "hin",
    "Croatian": "hrv",
    "Hungarian": "hun",
    "Armenian": "hye",
    "Indonesian": "ind",
    "Icelandic": "isl",
    "Italian": "ita",
    "Japanese": "jpn",
    "Javanese": "jav",
    "Georgian": "kat",
    "Kazakh": "kaz",
    "Khmer": "khm",
    "Kannada": "kan",
    "Korean": "kor",
    "Kurdish": "kur",
    "Kyrgyz": "kir",
    "Latin": "lat",
    "Lao": "lao",
    "Lithuanian": "lit",
    "Latvian": "lav",
    "Malagasy": "mlg",
    "Macedonian": "mkd",
    "Malayalam": "mal",
    "Mongolian": "mon",
    "Marathi": "mar",
    "Malay": "msa",
    "Burmese": "mya",
    "Nepali": "nep",
    "Dutch": "nld",
    "Norwegian": "nor",
    "Oromo": "orm",
    "Oriya": "ori",
    "Punjabi": "pan",
    "Polish": "pol",
    "Pashto": "pus",
    "Portuguese": "por",
    "Romanian": "ron",
    "Russian": "rus",
    "Sanskrit": "san",
    "Sindhi": "snd",
    "Sinhala": "sin",
    "Slovak": "slk",
    "Slovenian": "slv",
    "Somali": "som",
    "Albanian": "sqi",
    "Serbian": "srp",
    "Sundanese": "sun",
    "Swedish": "swe",
    "Swahili": "swa",
    "Tamil": "tam",
    "Telugu": "tel",
    "Thai": "tha",
    "Tagalog": "tgl",
    "Turkish": "tur",
    "Uyghur": "uig",
    "Ukrainian": "ukr",
    "Urdu": "urd",
    "Uzbek": "uzb",
    "Vietnamese": "vie",
    "Xhosa": "xho",
    "Yiddish": "yid",
    "Chinese": "chi_sim",
}

TESSERACT_CODE_TO_LANGUAGE = {v: k for k, v in LANGUAGE_TO_TESSERACT_CODE.items()}

# --- MoA Functions ---
def generate_together(
    model,
    messages,
    max_tokens=2048,
    temperature=0.1,
    streaming=False,
):
    output = None
    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            endpoint = "https://api.groq.com/openai/v1/chat/completions"
            if DEBUG:
                st.write(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`."
                )
            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature if temperature > 1e-4 else 0,
                    "messages": messages,
                },
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            )
            if "error" in res.json():
                st.write(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    st.write("Input + output is longer than max_position_id.")
                    return None
            output = res.json()["choices"][0]["message"]["content"]
            break
        except Exception as e:
            st.write(e)
            if DEBUG:
                st.write(f"Msgs: `{messages}`")
            st.write(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)
    if output is None:
        return output
    output = output.strip()
    if DEBUG:
        st.write(f"Output: `{output[:20]}...`.")
    return output


def generate_together_stream(model, messages, max_tokens=2048, temperature=0.7):
    endpoint = "https://api.groq.com/openai/v1/"
    client = openai.OpenAI(api_key=GROQ_API_KEY, base_url=endpoint)
    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        stream=True,  # this time, we set stream=True
    )
    return response


def generate_openai(model, messages, max_tokens=2048, temperature=0.1):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            if DEBUG:
                st.write(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break
        except Exception as e:
            st.write(e)
            st.write(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)
    output = output.strip()
    return output


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
    model,
    messages,
    references=[],
    max_tokens=2048,
    temperature=0.1,
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

# --- PDF Conversion Functions --- (from marker.convert)
def convert_single_pdf(
        fname: str,
        model_lst: List,
        max_pages: int = None,
        start_page: int = None,
        metadata: Optional[Dict] = None,
        langs: Optional[List[str]] = None,
        batch_multiplier: int = 1
) -> Tuple[str, Dict[str, Image.Image], Dict]:
    # Set language needed for OCR
    if langs is None:
        langs = [settings.DEFAULT_LANG]

    if metadata:
        langs = metadata.get("languages", langs)

    langs = replace_langs_with_codes(langs)
    validate_langs(langs)

    # Find the filetype
    filetype = find_filetype(fname)

    # Setup output metadata
    out_meta = {
        "languages": langs,
        "filetype": filetype,
    }

    if filetype == "other":  # We can't process this file
        return "", {}, out_meta

    # Get initial text blocks from the pdf
    doc = pdfium.PdfDocument(fname)
    pages, toc = get_text_blocks(
        doc,
        fname,
        max_pages=max_pages,
        start_page=start_page
    )
    out_meta.update({
        "toc": toc,
        "pages": len(pages),
    })

    # Trim pages from doc to align with start page
    if start_page:
        for page_idx in range(start_page):
            doc.del_page(0)

    # Unpack models from list
    texify_model, layout_model, order_model, edit_model, detection_model, ocr_model = model_lst

    # Identify text lines on pages
    surya_detection(doc, pages, detection_model, batch_multiplier=batch_multiplier)
    flush_cuda_memory()

    # OCR pages as needed
    pages, ocr_stats = run_ocr(doc, pages, langs, ocr_model, batch_multiplier=batch_multiplier)
    flush_cuda_memory()

    out_meta["ocr_stats"] = ocr_stats
    if len([b for p in pages for b in p.blocks]) == 0:
        print(f"Could not extract any text blocks for {fname}")
        return "", {}, out_meta

    surya_layout(doc, pages, layout_model, batch_multiplier=batch_multiplier)
    flush_cuda_memory()

    # Find headers and footers
    bad_span_ids = filter_header_footer(pages)
    out_meta["block_stats"] = {"header_footer": len(bad_span_ids)}

    # Add block types in
    annotate_block_types(pages)

    # Dump debug data if flags are set
    dump_bbox_debug_data(doc, fname, pages)

    # Find reading order for blocks
    # Sort blocks by reading order
    surya_order(doc, pages, order_model, batch_multiplier=batch_multiplier)
    sort_blocks_in_reading_order(pages)
    flush_cuda_memory()

    # Fix code blocks
    code_block_count = identify_code_blocks(pages)
    out_meta["block_stats"]["code"] = code_block_count
    indent_blocks(pages)

    # Fix table blocks
    table_count = format_tables(pages)
    out_meta["block_stats"]["table"] = table_count

    for page in pages:
        for block in page.blocks:
            block.filter_spans(bad_span_ids)
            block.filter_bad_span_types()

    filtered, eq_stats = replace_equations(
        doc,
        pages,
        texify_model,
        batch_multiplier=batch_multiplier
    )
    flush_cuda_memory()
    out_meta["block_stats"]["equations"] = eq_stats

    # Extract images and figures
    if settings.EXTRACT_IMAGES:
        extract_images(doc, pages)

    # Split out headers
    split_heading_blocks(pages)
    find_bold_italic(pages)

    # Copy to avoid changing original data
    merged_lines = merge_spans(filtered)
    text_blocks = merge_lines(merged_lines)
    text_blocks = filter_common_titles(text_blocks)
    full_text = get_full_text(text_blocks)

    # Handle empty blocks being joined
    full_text = cleanup_text(full_text)

    # Replace bullet characters with a -
    full_text = replace_bullets(full_text)

    # Postprocess text with editor model
    full_text, edit_stats = edit_full_text(
        full_text,
        edit_model,
        batch_multiplier=batch_multiplier
    )
    flush_cuda_memory()
    out_meta["postprocess_stats"] = {"edit": edit_stats}
    doc_images = images_to_dict(pages)

    return full_text, doc_images, out_meta

# --- Other functions from marker --- (If required, include functions like
# markdown_exists, save_markdown, flush_cuda_memory from marker.output and marker.utils)

# --- Streamlit App ---
st.title("PDF to Markdown Converter & Report Generator")

# --- Sidebar ---
st.sidebar.header("Options")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

# Language selection for OCR
selected_languages = st.sidebar.multiselect(
    'Select Languages for OCR',
    list(LANGUAGE_TO_TESSERACT_CODE.keys()),
    ['English'] # Default language
)

model_name = st.sidebar.selectbox(
    "Select Model",
    default_reference_models,
    help="Select a model for report generation."
)

# --- Main Content Area ---
# 1. PDF Conversion and Storage
if uploaded_files:
    # Load models
    model_lst = load_all_models(langs=selected_languages)
    pdf_content = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for uploaded_file in uploaded_files:
            with open(os.path.join(tmpdir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner(f"Converting {uploaded_file.name}..."):
                full_text, images, out_metadata = convert_single_pdf(
                    os.path.join(tmpdir, uploaded_file.name), model_lst, langs=selected_languages
                )
            pdf_content[uploaded_file.name] = full_text

            # Display converted Markdown and images
            st.markdown(f"## {uploaded_file.name} (Markdown)")
            st.markdown(full_text)
            st.markdown(f"### Images from {uploaded_file.name}")
            for filename, image in images.items():
                st.image(image, caption=filename)

    # 2. Report Generation
    st.header("Report Generation")
    report_topic = st.text_input("Enter the topic for your report:")
    if st.button("Generate Report"):
        if report_topic:
            with st.spinner("Generating report..."):
                references = list(pdf_content.values())
                messages = [
                    {"role": "user", "content": f"Create a structured report about {report_topic}, using the provided reference documents. Ensure to include citations and references."},
                ]
                report = generate_with_references(
                    model=model_name, messages=messages, references=references
                )
            st.markdown(report)

            # 3. Report Modification
            st.header("Report Modification")
            modification_request = st.text_input("Enter your modification request (e.g., 'expand on XYZ topic'):")
            if st.button("Modify Report"):
                if modification_request:
                    with st.spinner("Modifying report..."):
                        messages = [
                            {"role": "user", "content": f"Here's the current report:\n\n{report}\n\n{modification_request}"}
                        ]
                        modified_report = generate_with_references(
                            model=model_name, messages=messages, references=references
                        )
                    st.markdown(modified_report)
        else:
            st.warning("Please enter a report topic.")
else:
    st.warning("Please upload at least one PDF file.")

# --- Clean up CUDA Memory ---
flush_cuda_memory()
