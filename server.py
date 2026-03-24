import os
import io
import re
import json
import uuid
import time
import base64
import hashlib
import asyncio
import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, HTMLResponse, Response
from mcp.server.sse import SseServerTransport
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account
from pypdf import PdfReader, PdfWriter
from pydantic import BaseModel, Field

load_dotenv()

UPLOAD_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" type="image/png" href="/favicon-32.png">
<title>Document AI Parser</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; min-height: 100vh;
         display: flex; align-items: center; justify-content: center; }
  .container { max-width: 600px; width: 100%; padding: 2rem; }
  h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
  p.sub { color: #94a3b8; margin-bottom: 1.5rem; font-size: 0.9rem; }
  .drop-zone { border: 2px dashed #475569; border-radius: 12px; padding: 3rem 2rem;
                text-align: center; cursor: pointer; transition: all 0.2s; }
  .drop-zone:hover, .drop-zone.drag-over { border-color: #3b82f6; background: #1e293b; }
  .drop-zone input { display: none; }
  .drop-zone p { color: #94a3b8; }
  .drop-zone .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
  #status { margin-top: 1.5rem; }
  .parsing { color: #fbbf24; }
  .success { background: #1e293b; border-radius: 8px; padding: 1rem; margin-top: 1rem; }
  .doc-id { font-family: monospace; font-size: 1.2rem; color: #34d399;
            background: #0f172a; padding: 0.3rem 0.6rem; border-radius: 4px;
            user-select: all; cursor: pointer; }
  .copy-btn { background: #3b82f6; color: white; border: none; padding: 0.4rem 1rem;
              border-radius: 6px; cursor: pointer; margin-left: 0.5rem; font-size: 0.85rem; }
  .copy-btn:hover { background: #2563eb; }
  .instructions { color: #94a3b8; font-size: 0.85rem; margin-top: 0.75rem; line-height: 1.5; }
  .error { color: #f87171; margin-top: 1rem; }
  .parser-toggle { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; }
  .parser-toggle button { flex: 1; padding: 0.6rem; border: 2px solid #475569; border-radius: 8px;
                           background: transparent; color: #94a3b8; cursor: pointer; font-size: 0.9rem;
                           transition: all 0.2s; }
  .parser-toggle button.active { border-color: #3b82f6; background: #1e293b; color: #e2e8f0; }
  .parser-toggle button:hover { border-color: #3b82f6; }
</style>
</head>
<body>
<div class="container">
  <h1>Document AI Parser</h1>
  <p class="sub">Upload a PDF or image to parse and compare OCR engines</p>
  <div class="parser-toggle">
    <button id="btnMistral" class="active" onclick="setParser('mistral')">Mistral OCR</button>
    <button id="btnGoogle" onclick="setParser('google')">Google Document AI</button>
  </div>
  <div class="drop-zone" id="dropZone">
    <div class="icon">📄</div>
    <p>Drag & drop a file here, or click to browse</p>
    <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg,.tiff,.gif,.bmp,.webp">
  </div>
  <div id="status"></div>
</div>
<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const status = document.getElementById('status');
let currentParser = 'mistral';

function setParser(p) {
  currentParser = p;
  document.getElementById('btnGoogle').classList.toggle('active', p === 'google');
  document.getElementById('btnMistral').classList.toggle('active', p === 'mistral');
}

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) uploadFile(fileInput.files[0]); });

async function uploadFile(file) {
  const parserName = currentParser === 'mistral' ? 'Mistral OCR' : 'Google Document AI';
  status.innerHTML = '<p class="parsing">Parsing ' + file.name + ' with ' + parserName + '...</p>';
  const form = new FormData();
  form.append('file', file);
  try {
    const resp = await fetch('/parse?parser=' + currentParser, { method: 'POST', body: form });
    const data = await resp.json();
    if (data.error) { status.innerHTML = '<p class="error">Error: ' + data.error + '</p>'; return; }
    const usedParser = data.parser || currentParser;
    status.innerHTML = `
      <div class="success">
        <p>Parsed <strong>${data.filename}</strong> via <strong>${usedParser}</strong> (${data.pages} pages, ${data.chars.toLocaleString()} chars)</p>
        <p style="margin-top:0.75rem">Document ID: <span class="doc-id">${data.document_id}</span>
          <button class="copy-btn" onclick="navigator.clipboard.writeText('${data.document_id}')">Copy</button></p>
        <p class="instructions">
          Paste this into Claude chat:<br>
          <em>"Use the parsing service to get parsed result for document <strong>${data.document_id}</strong>"</em>
        </p>
      </div>`;
  } catch (e) { status.innerHTML = '<p class="error">Upload failed: ' + e.message + '</p>'; }
}
</script>
</body>
</html>"""

# Google Document AI config
GCP_PROJECT_ID = os.environ.get("GOOGLE_DOCAI_PROJECT_ID", "decoded-flag-490415-n5")
GCP_LOCATION = os.environ.get("GOOGLE_DOCAI_LOCATION", "us")
GCP_PROCESSOR_ID = os.environ.get("GOOGLE_DOCAI_PROCESSOR_ID", "8a96e920607e3974")

# Service account credentials — base64-encoded JSON string
GOOGLE_DOCAI_CREDENTIALS_PATH = os.environ.get("GOOGLE_DOCAI_CREDENTIALS_PATH", "")

# Mistral OCR config
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-ocr-latest")


def _get_docai_client():
    """Create a Document AI client with proper credentials."""
    endpoint = f"{GCP_LOCATION}-documentai.googleapis.com"
    if GOOGLE_DOCAI_CREDENTIALS_PATH:
        decoded = base64.b64decode(GOOGLE_DOCAI_CREDENTIALS_PATH)
        info = json.loads(decoded)
        creds = service_account.Credentials.from_service_account_info(info)
        return documentai.DocumentProcessorServiceClient(
            credentials=creds,
            client_options={"api_endpoint": endpoint},
        )
    else:
        return documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": endpoint},
        )


CHUNK_SIZE = 1  # Send 1 page at a time — guarantees correct page numbering
MAX_CONCURRENT = 175  # Max parallel requests to Document AI


def _split_pdf(pdf_bytes: bytes) -> list[bytes]:
    """Split a PDF into chunks of CHUNK_SIZE pages. Returns list of PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)

    if total_pages <= CHUNK_SIZE:
        return [pdf_bytes]

    chunks = []
    for start in range(0, total_pages, CHUNK_SIZE):
        writer = PdfWriter()
        for page_num in range(start, min(start + CHUNK_SIZE, total_pages)):
            writer.add_page(reader.pages[page_num])
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append(buf.getvalue())

    return chunks


def _extract_tables_from_page(page, document_text: str) -> list[str]:
    """Extract tables from a page as markdown-formatted tables."""
    tables = []
    for table in page.tables:
        rows = []
        for header_row in table.header_rows:
            cells = []
            for cell in header_row.cells:
                text = _get_text_from_layout(cell.layout, document_text).strip()
                cells.append(text)
            rows.append("| " + " | ".join(cells) + " |")
            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        for body_row in table.body_rows:
            cells = []
            for cell in body_row.cells:
                text = _get_text_from_layout(cell.layout, document_text).strip()
                cells.append(text)
            rows.append("| " + " | ".join(cells) + " |")
        if rows:
            tables.append("\n".join(rows))
    return tables


def _get_text_from_layout(layout, document_text: str) -> str:
    """Extract text from a layout element using text anchors."""
    text = ""
    for segment in layout.text_anchor.text_segments:
        start = int(segment.start_index) if segment.start_index else 0
        end = int(segment.end_index)
        text += document_text[start:end]
    return text


def _get_page_text(page, document_text: str) -> str:
    """Extract the text content for a single page using its layout text anchor."""
    if page.layout and page.layout.text_anchor and page.layout.text_anchor.text_segments:
        parts = []
        for segment in page.layout.text_anchor.text_segments:
            start = int(segment.start_index) if segment.start_index else 0
            end = int(segment.end_index)
            parts.append(document_text[start:end])
        return "".join(parts)
    return ""


def _process_single_chunk(file_content: bytes, mime_type: str, page_offset: int = 0) -> str:
    """Send a single document chunk to Document AI and return parsed text with page markers and tables."""
    client = _get_docai_client()
    resource_name = client.processor_path(GCP_PROJECT_ID, GCP_LOCATION, GCP_PROCESSOR_ID)
    raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)

    # Layout Parser with Gemini — uses layout_config, not ocr_config
    process_options = documentai.ProcessOptions(
        layout_config=documentai.ProcessOptions.LayoutConfig(
            chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
                chunk_size=1000,
                include_ancestor_headings=True,
            ),
        ),
    )

    request = documentai.ProcessRequest(
        name=resource_name,
        raw_document=raw_document,
        process_options=process_options,
    )
    result = client.process_document(request=request)
    document = result.document

    # Try page-level extraction first (works with OCR processor)
    has_page_text = False
    if document.text and document.pages:
        for page in document.pages:
            if _get_page_text(page, document.text).strip():
                has_page_text = True
                break

    if has_page_text:
        # OCR-style extraction: pages with text anchors and tables
        output_parts = []
        for i, page in enumerate(document.pages):
            page_num = page_offset + (page.page_number if page.page_number else i + 1)
            page_text = _get_page_text(page, document.text).strip()

            output_parts.append(f"[Page {page_num}]")
            if page_text:
                output_parts.append(page_text)

            tables = _extract_tables_from_page(page, document.text)
            for j, table_md in enumerate(tables):
                output_parts.append(f"[Table {j+1} on page {page_num}]\n{table_md}")

        return "\n\n".join(output_parts)

    # Layout Parser extraction: chunks are the structured output
    # With CHUNK_SIZE=1, each API call is one page, so all chunks belong to page_offset+1
    if document.chunked_document and document.chunked_document.chunks:
        page_num = page_offset + 1
        chunk_texts = []
        for chunk in document.chunked_document.chunks:
            content = chunk.content.strip() if chunk.content else ""
            if content:
                chunk_texts.append(content)

        if chunk_texts:
            return f"[Page {page_num}]\n\n" + "\n\n".join(chunk_texts)
        return ""

    # Fallback: raw document text with page markers from document.text
    if document.text:
        output_parts = []
        for i, page in enumerate(document.pages):
            page_num = page_offset + (page.page_number if page.page_number else i + 1)
            output_parts.append(f"[Page {page_num}]")
        if output_parts:
            output_parts.append(document.text)
        return "\n\n".join(output_parts)

    return ""


async def _process_document(file_content: bytes, mime_type: str = "application/pdf") -> str:
    """Process a document, splitting large PDFs into parallel chunks."""
    if mime_type == "application/pdf":
        chunks = _split_pdf(file_content)
    else:
        chunks = [file_content]

    if len(chunks) == 1:
        return await asyncio.to_thread(_process_single_chunk, chunks[0], mime_type, 0)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def process_with_limit(chunk: bytes, offset: int) -> str:
        async with semaphore:
            return await asyncio.to_thread(_process_single_chunk, chunk, mime_type, offset)

    tasks = [
        process_with_limit(c, i * CHUNK_SIZE)
        for i, c in enumerate(chunks)
    ]
    results = await asyncio.gather(*tasks)
    return "\n\n".join(results)


# --- Mistral OCR processing ---

def _get_mistral_client():
    """Create a Mistral client."""
    from mistralai.client import Mistral
    return Mistral(api_key=MISTRAL_API_KEY)


class _ImageAnnotation(BaseModel):
    """Schema for Mistral bbox annotation — describes each extracted image."""
    image_type: str = Field(..., description="Type of image: chart, table, logo, photo, diagram, figure, signature, or other")
    description: str = Field(..., description="Extract ALL visible text, numbers, labels, and values from the image. For charts and infographics, list every metric with its exact value. Be comprehensive — do not summarize or abbreviate.")


async def _process_document_mistral(file_bytes: bytes, mime_type: str = "application/pdf") -> str:
    """Process a document using Mistral OCR. Supports up to 1000 pages / 50MB."""
    from mistralai.client import models as mistral_models
    from mistralai.extra import response_format_from_pydantic_model

    client = _get_mistral_client()

    # Step 1: Upload file to Mistral
    filename = "document.pdf" if "pdf" in mime_type else "document.png"
    uploaded = await asyncio.to_thread(
        client.files.upload,
        file=mistral_models.File(file_name=filename, content=file_bytes),
        purpose="ocr",
    )

    # Step 2: Run OCR with full extraction settings + image annotations
    ocr_response = await asyncio.to_thread(
        client.ocr.process,
        model=MISTRAL_MODEL,
        document=mistral_models.FileChunk(file_id=uploaded.id, type="file"),
        table_format="markdown",
        extract_header=True,
        extract_footer=True,
        include_image_base64=False,
        bbox_annotation_format=response_format_from_pydantic_model(_ImageAnnotation),
    )

    # Step 3: Convert response to [Page N] format matching Google output
    output_parts = []
    for page in ocr_response.pages:
        page_num = page.index + 1  # Mistral uses 0-indexed
        parts = [f"[Page {page_num}]"]

        if page.header:
            parts.append(f"[Header]\n{page.header}")

        # Inline table content (replace [tbl-X](tbl-X) placeholders)
        md = page.markdown or ""
        if page.tables:
            for table in page.tables:
                placeholder = f"[{table.id}]({table.id})"
                md = md.replace(placeholder, f"\n{table.content}\n")

        # Append image annotations after their placeholders
        if page.images:
            for img in page.images:
                placeholder = f"![{img.id}]({img.id})"
                if img.image_annotation and placeholder in md:
                    try:
                        ann = json.loads(img.image_annotation) if isinstance(img.image_annotation, str) else img.image_annotation
                        caption = f"\n[{ann.get('image_type', 'image')}: {ann.get('description', '')}]"
                    except (json.JSONDecodeError, TypeError):
                        caption = f"\n[image: {img.image_annotation}]"
                    md = md.replace(placeholder, placeholder + caption)

        if md.strip():
            parts.append(md)

        if page.footer:
            parts.append(f"[Footer]\n{page.footer}")

        output_parts.append("\n\n".join(parts))

    return "\n\n".join(output_parts)


# --- Parsed result cache + upload tokens ---
_parsed_results: dict[str, dict] = {}  # id -> {"text": str, "pages": int, "filename": str}
_content_hash_to_id: dict[str, str] = {}  # sha256 hash -> doc_id (for cache dedup)
_upload_tokens: dict[str, dict] = {}   # token -> {"created": float, "filename": str}
UPLOAD_TOKEN_TTL = 300  # 5 minutes


def _check_cache(file_bytes: bytes) -> str | None:
    """Return cached doc_id if this file content was already parsed, else None."""
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    return _content_hash_to_id.get(content_hash)


def _store_result(doc_id: str, file_bytes: bytes, text: str, pages: int, filename: str):
    """Store a parsed result and its content hash for dedup."""
    _parsed_results[doc_id] = {"text": text, "pages": pages, "filename": filename}
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    _content_hash_to_id[content_hash] = doc_id

from mcp.server.transport_security import TransportSecuritySettings

# --- Claude MCP instance (SSE transport at /sse) ---
mcp = FastMCP("BW Document OCR")


# --- ChatGPT MCP instance (streamable HTTP at /mcp) ---
_WIDGET_URI = f"ui://widget/parser-v3.html?v={int(time.time())}"
import mcp.types as mcp_types

mcp_chatgpt = FastMCP(
    "BW Document OCR",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp_chatgpt.resource(_WIDGET_URI, name="parser-widget", mime_type="text/html",
                       description="Bellwether Document Parser — drag and drop PDFs to extract text")
def get_parser_widget():
    widget_path = os.path.join(os.path.dirname(__file__), "public", "parser-widget.html")
    with open(widget_path, "r", encoding="utf-8") as f:
        return f.read()


@mcp_chatgpt.tool(
    meta={
        "openai/outputTemplate": _WIDGET_URI,
        "openai/toolInvocation/invoking": "Opening Bellwether Document Parser...",
        "openai/toolInvocation/invoked": "Parser ready — drop a file to parse",
    },
)
async def upload_and_parse() -> str:
    """Upload and parse a PDF or image document.

    Use this when the user wants to parse, OCR, or extract text from a document.
    Opens the Bellwether Document Parser where they can drag and drop files.
    """
    return "Bellwether Document Parser ready. Drag and drop a file to extract text."


@mcp_chatgpt.tool()
async def parse_from_url(url: str, filename: str = "") -> str:
    """Parse a document from a URL using OCR.

    Args:
        url: Direct URL to the document file
        filename: Optional original filename (used when URL doesn't contain a meaningful name)
    """
    if not MISTRAL_API_KEY:
        return "Error: No Mistral API key configured."
    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            file_bytes = resp.content
    except Exception as e:
        return f"Error downloading: {str(e)}"

    if not filename:
        filename = url.split("/")[-1].split("?")[0] or "document.pdf"
    mime_type = "application/pdf"
    lower = filename.lower()
    if lower.endswith(".png"): mime_type = "image/png"
    elif lower.endswith((".jpg", ".jpeg")): mime_type = "image/jpeg"

    cached_id = _check_cache(file_bytes)
    if cached_id:
        entry = _parsed_results[cached_id]
        return json.dumps({"document_id": cached_id, "filename": entry["filename"], "pages": entry["pages"], "chars": len(entry["text"])})

    try:
        result = await _process_document_mistral(file_bytes, mime_type)
    except Exception as e:
        return f"Error: {str(e)}"

    page_count = 1
    if mime_type == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            page_count = len(reader.pages)
        except Exception:
            pass

    doc_id = str(uuid.uuid4())[:8]
    _store_result(doc_id, file_bytes, result, page_count, filename)
    return json.dumps({"document_id": doc_id, "filename": filename, "pages": page_count, "chars": len(result)})


@mcp_chatgpt.tool()
async def parse_from_base64(data: str, filename: str = "document.pdf") -> str:
    """Parse a document from base64-encoded file data.

    Args:
        data: Base64-encoded file content
        filename: Original filename (used to detect mime type)
    """
    if not MISTRAL_API_KEY:
        return "Error: No Mistral API key configured."
    try:
        file_bytes = base64.b64decode(data)
    except Exception as e:
        return f"Error decoding base64: {str(e)}"

    mime_type = "application/pdf"
    lower = filename.lower()
    if lower.endswith(".png"): mime_type = "image/png"
    elif lower.endswith((".jpg", ".jpeg")): mime_type = "image/jpeg"
    elif lower.endswith(".tiff"): mime_type = "image/tiff"

    cached_id = _check_cache(file_bytes)
    if cached_id:
        entry = _parsed_results[cached_id]
        return json.dumps({"document_id": cached_id, "filename": entry["filename"], "pages": entry["pages"], "chars": len(entry["text"])})

    try:
        result = await _process_document_mistral(file_bytes, mime_type)
    except Exception as e:
        return f"Error: {str(e)}"

    page_count = 1
    if mime_type == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            page_count = len(reader.pages)
        except Exception:
            pass

    doc_id = str(uuid.uuid4())[:8]
    _store_result(doc_id, file_bytes, result, page_count, filename)
    return json.dumps({"document_id": doc_id, "filename": filename, "pages": page_count, "chars": len(result)})


@mcp_chatgpt.tool()
async def get_parsed_file(document_id: str) -> list:
    """Get parsed document as a downloadable markdown file.

    Args:
        document_id: The document ID from a previous parse
    """
    if document_id not in _parsed_results:
        return [mcp_types.TextContent(type="text", text=f"No document with id '{document_id}'.")]

    entry = _parsed_results[document_id]
    original_name = entry.get("filename", "document")
    base_name = original_name.rsplit(".", 1)[0] if "." in original_name else original_name
    md_filename = f"{base_name}.md"
    content_b64 = base64.b64encode(entry["text"].encode("utf-8")).decode("utf-8")

    return [mcp_types.EmbeddedResource(
        type="resource",
        resource=mcp_types.TextResourceContents(
            uri=f"data:text/markdown;base64,{content_b64}",
            mimeType="text/markdown",
            text=entry["text"],
        ),
    )]


@mcp.tool()
async def get_parsed_result(
    document_id: str,
    page_start: int = 1,
    page_end: int = 0,
    output_format: str = "markdown",
) -> str:
    """Get the parsed text content of a document by its ID.

    Use this when the user provides a document_id from a previous parse,
    or after calling parse_document_from_url. Supports page ranges for large documents.

    Args:
        document_id: The document ID (e.g. "4a0907f7")
        page_start: First page to return (1-indexed, default: 1)
        page_end: Last page to return (0 = all remaining pages)
        output_format: "markdown" (default), "plain" (strip markers/HTML), or "tables_only" (just tables)
    """
    if document_id not in _parsed_results:
        return (
            f"Error: No parsed document with id '{document_id}'. "
            "Upload and parse a file first via POST /parse endpoint or provide a URL."
        )

    entry = _parsed_results[document_id]
    full_text = entry["text"]
    total_pages = entry["pages"]

    # Page range extraction
    if page_start > 1 or page_end > 0:
        page_sections = re.split(r'(?=\[Page \d+\])', full_text)
        page_sections = [s for s in page_sections if s.strip()]
        page_map = {}
        for section in page_sections:
            match = re.match(r'\[Page (\d+)\]', section)
            if match:
                page_map[int(match.group(1))] = section
        end = page_end if page_end > 0 else total_pages
        selected_parts = [page_map[p] for p in range(page_start, end + 1) if p in page_map]
        text = "\n\n".join(selected_parts) if selected_parts else "(No content for requested pages)"
        header = f"[Document: {entry['filename']}, pages {page_start}-{end} of {total_pages}]"
    else:
        text = full_text
        header = f"[Document: {entry['filename']}, {total_pages} pages]"

    # Format output
    if output_format == "plain":
        # Strip [Page N] markers, [Header], [Footer], and HTML tags
        text = re.sub(r'\[Page \d+\]', '', text)
        text = re.sub(r'\[(Header|Footer)\]', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
    elif output_format == "tables_only":
        # Extract only table content (HTML tables or markdown tables)
        tables = re.findall(r'(<table.*?</table>)', text, re.DOTALL)
        md_tables = re.findall(r'(\|.+\|(?:\n\|.+\|)+)', text)
        all_tables = tables + md_tables
        if all_tables:
            text = "\n\n---\n\n".join(all_tables)
        else:
            text = "(No tables found in this document)"

    return f"{header}\n\n{text}"


@mcp.tool()
async def get_download_link(
    document_id: str,
) -> str:
    """Get a download link for a parsed document as a markdown file.

    Use this when the user says "download", "give me the file", "export as markdown",
    or "save as md". Returns a direct download URL — present it as a clickable link.
    Do NOT read or summarize the file contents. Just give the user the link.

    Args:
        document_id: The document ID (e.g. "4a0907f7")
    """
    if document_id not in _parsed_results:
        return f"Error: No document with id '{document_id}'."

    entry = _parsed_results[document_id]
    base_name = entry["filename"].rsplit(".", 1)[0] if "." in entry["filename"] else entry["filename"]

    return (
        f"Download link for {entry['filename']} ({entry['pages']} pages, {len(entry['text']):,} chars):\n\n"
        f"{UPLOAD_URL}/result-file/{document_id}\n\n"
        f"Give this link to the user as a clickable download. Do NOT read or summarize the file."
    )


async def list_documents() -> str:
    """List all documents that have been parsed and are available in the cache.

    Use this when the user asks "what documents have I parsed?", "show my documents",
    or needs to find a document_id they forgot.
    """
    if not _parsed_results:
        return f"No documents parsed yet. Upload a file at {UPLOAD_URL} or provide a URL to parse."

    lines = [f"Parsed documents ({len(_parsed_results)}):\n"]
    for doc_id, entry in _parsed_results.items():
        lines.append(f"  {doc_id} — {entry['filename']} ({entry['pages']} pages, {len(entry['text']):,} chars)")

    lines.append(f"\nTo retrieve text: call get_parsed_result(document_id)")
    return "\n".join(lines)


async def get_upload_url(
    filename: str = "document.pdf",
) -> str:
    """Internal: generate a one-time upload URL. Used by the web UI, not exposed as MCP tool."""
    # Clean expired tokens
    now = time.time()
    expired = [t for t, v in _upload_tokens.items() if now - v["created"] > UPLOAD_TOKEN_TTL]
    for t in expired:
        del _upload_tokens[t]

    token = str(uuid.uuid4())
    _upload_tokens[token] = {"created": now, "filename": filename}

    upload_url = f"{UPLOAD_URL}/upload/{token}"

    return (
        f"Upload URL ready (expires in 5 minutes).\n\n"
        f"Upload with curl:\n"
        f'curl -s -X POST "{upload_url}" -F "file=@FILE_PATH"\n\n'
        f"The response JSON contains a document_id. "
        f"Call get_parsed_result(document_id) to retrieve the parsed text."
    )


@mcp.tool()
async def parse_document_from_url(
    url: str,
    mime_type: str = "application/pdf",
    parser: str = "mistral",
) -> str:
    """Parse a PDF, image, or document from a URL using OCR.

    Use this when the user says "parse this PDF", "OCR this document", "extract text
    from this file", or provides a URL to a document. Supports PDF, PNG, JPG, TIFF.
    Returns the full extracted text with page markers.

    Args:
        url: Direct URL to a document file
        mime_type: MIME type of the document (default: application/pdf)
        parser: Which OCR engine to use: "mistral" (default) or "google"
    """
    if parser == "mistral":
        if not MISTRAL_API_KEY:
            return "Error: No Mistral API key configured. Set MISTRAL_API_KEY."
    elif not GOOGLE_DOCAI_CREDENTIALS_PATH:
        return "Error: No Google Cloud credentials configured. Set GOOGLE_DOCAI_CREDENTIALS_PATH."

    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            file_bytes = resp.content
    except Exception as e:
        return f"Error downloading document from URL: {str(e)}"

    # Auto-detect mime type from URL extension
    lower_url = url.lower().split("?")[0]
    if lower_url.endswith(".png"):
        mime_type = "image/png"
    elif lower_url.endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif lower_url.endswith(".tiff"):
        mime_type = "image/tiff"

    filename = url.split("/")[-1].split("?")[0] or "document.pdf"

    # Check cache first
    cached_id = _check_cache(file_bytes)
    if cached_id:
        entry = _parsed_results[cached_id]
        result = entry["text"]
        page_count = entry["pages"]
        if len(result) < 50000:
            return f"[Cached: {entry['filename']}, {page_count} pages, id: {cached_id}]\n\n{result}"
        else:
            return (
                f"[Cached: {entry['filename']}, {page_count} pages, {len(result):,} chars, id: {cached_id}]\n\n"
                f"Document is large. Use get_parsed_result(document_id='{cached_id}') to retrieve the full text.\n"
                f"For specific pages: get_parsed_result('{cached_id}', page_start=1, page_end=5)\n"
                f"Recommended: fetch 5-10 pages at a time for large documents.\n"
                f"For a downloadable .md file: call get_download_link(document_id)"
            )

    try:
        if parser == "mistral":
            result = await _process_document_mistral(file_bytes, mime_type)
        else:
            result = await _process_document(file_bytes, mime_type)

        # Count pages
        page_count = 0
        if mime_type == "application/pdf":
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                page_count = len(reader.pages)
            except Exception:
                page_count = 1
        else:
            page_count = 1

        # Cache the result for later retrieval via get_parsed_result
        parser_label = "mistral" if parser == "mistral" else "google"
        doc_id = str(uuid.uuid4())[:8]
        _store_result(doc_id, file_bytes, result, page_count, filename)

        # Small docs: return inline. Large docs: return metadata + hints for chunked retrieval
        if len(result) < 50000:
            return f"[Parsed {filename} via {parser_label}, {page_count} pages, id: {doc_id}]\n\n{result}"
        else:
            return (
                f"[Parsed {filename} via {parser_label}, {page_count} pages, {len(result):,} chars, id: {doc_id}]\n\n"
                f"Document is large. Use get_parsed_result(document_id='{doc_id}') to retrieve the full text.\n"
                f"For specific pages: get_parsed_result('{doc_id}', page_start=1, page_end=5)\n"
                f"Recommended: fetch 5-10 pages at a time for large documents.\n"
                f"For a downloadable .md file: call get_download_link(document_id)"
            )
    except Exception as e:
        return f"Error: {str(e)}"


UPLOAD_URL = os.environ.get("PUBLIC_URL", "https://bw-parse-mcp-server.up.railway.app")


async def check_status() -> str:
    """Check parser status and see what documents have been parsed.

    Call this to see available parsers, cached documents, and the upload URL.
    """
    lines = ["Bellwether Document Parser is running.\n"]

    if MISTRAL_API_KEY:
        lines.append(f"Mistral OCR: ready (model: {MISTRAL_MODEL})")
    else:
        lines.append("Mistral OCR: NOT configured")

    if GOOGLE_DOCAI_CREDENTIALS_PATH:
        lines.append("Google Document AI: ready")
    else:
        lines.append("Google Document AI: NOT configured")

    # Show cached documents
    if _parsed_results:
        lines.append(f"\nParsed documents in cache ({len(_parsed_results)}):")
        for doc_id, entry in _parsed_results.items():
            lines.append(f"  {doc_id} — {entry['filename']} ({entry['pages']} pages, {len(entry['text'])} chars)")
    else:
        lines.append("\nNo documents parsed yet.")

    lines.append(
        f"\nHow to parse:\n"
        f"- From URL: call parse_document_from_url(url)\n"
        f"- Upload manually: {UPLOAD_URL}\n"
        f"- Retrieve text: call get_parsed_result(document_id)"
    )

    return "\n".join(lines)


# --- Starlette app with SSE transport + HTTP endpoints ---

sse = SseServerTransport("/messages/")


async def handle_sse(request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as (read_stream, write_stream):
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options(),
        )


async def homepage(request):
    """Upload page — only accessible with ?key= param."""
    access_key = os.environ.get("UPLOAD_ACCESS_KEY", "")
    if access_key and request.query_params.get("key") != access_key:
        return PlainTextResponse("BW Document OCR is running. Access the parser via Claude MCP or ChatGPT.", status_code=200)
    return HTMLResponse(UPLOAD_PAGE_HTML)


async def favicon(request):
    """Serve the favicon."""
    favicon_path = os.path.join(os.path.dirname(__file__), "favicon-32.png")
    with open(favicon_path, "rb") as f:
        return Response(f.read(), media_type="image/png")


async def health(request):
    return JSONResponse({"status": "ok"})


async def parse_endpoint(request: Request):
    """Upload + parse in one step. Returns a document_id for MCP tool retrieval.
    Pass ?parser=google to use Google Document AI instead of Mistral OCR."""
    parser = request.query_params.get("parser", "mistral")

    if parser == "mistral":
        if not MISTRAL_API_KEY:
            return JSONResponse({"error": "No Mistral API key configured"}, status_code=500)
    elif not GOOGLE_DOCAI_CREDENTIALS_PATH:
        return JSONResponse({"error": "No credentials configured"}, status_code=500)

    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file")
        if not file:
            return JSONResponse({"error": "No file field in form"}, status_code=400)
        file_bytes = await file.read()
        filename = file.filename or "document.pdf"
        mime_type = file.content_type or "application/pdf"
        # Allow parser override from form field too
        if form.get("parser"):
            parser = form["parser"]
    elif "application/json" in content_type:
        # ChatGPT Actions may send JSON with a URL or openaiFileIdRefs
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        # If ChatGPT sent a URL in the JSON body, download it
        url = body.get("url") or body.get("file_url") or body.get("document_url")
        if url:
            try:
                async with httpx.AsyncClient(timeout=120, follow_redirects=True) as dl_client:
                    resp = await dl_client.get(url)
                    resp.raise_for_status()
                    file_bytes = resp.content
            except Exception as e:
                return JSONResponse({"error": f"Failed to download file: {str(e)}"}, status_code=400)
            filename = url.split("/")[-1].split("?")[0] or "document.pdf"
            lower_url = url.lower().split("?")[0]
            if lower_url.endswith(".png"):
                mime_type = "image/png"
            elif lower_url.endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            else:
                mime_type = "application/pdf"
        else:
            return JSONResponse({
                "error": "JSON body received but no file URL found. Send file as multipart/form-data or provide a 'url' field in JSON."
            }, status_code=400)
        if body.get("parser"):
            parser = body["parser"]
    else:
        file_bytes = await request.body()
        filename = request.headers.get("x-filename", "document.pdf")
        mime_type = content_type if content_type and content_type != "application/octet-stream" else "application/pdf"

    if not file_bytes:
        return JSONResponse({"error": "Empty file"}, status_code=400)

    # Check cache first — skip re-parsing if same file was already processed
    cached_id = _check_cache(file_bytes)
    if cached_id:
        entry = _parsed_results[cached_id]
        return JSONResponse({
            "document_id": cached_id,
            "filename": entry["filename"],
            "pages": entry["pages"],
            "chars": len(entry["text"]),
            "cached": True,
            "message": f"Found cached result. Use get_parsed_result(document_id='{cached_id}') to retrieve the text.",
        })

    # Count pages
    page_count = 0
    if mime_type == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            page_count = len(reader.pages)
        except Exception:
            page_count = 1
    else:
        page_count = 1

    try:
        if parser == "mistral":
            result = await _process_document_mistral(file_bytes, mime_type)
        else:
            result = await _process_document(file_bytes, mime_type)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    parser_label = "mistral" if parser == "mistral" else "google"
    doc_id = str(uuid.uuid4())[:8]
    _store_result(doc_id, file_bytes, result, page_count, filename)

    return JSONResponse({
        "document_id": doc_id,
        "filename": filename,
        "pages": page_count,
        "chars": len(result),
        "parser": parser_label,
        "message": (
            f"Parsed successfully via {parser_label}. In Claude, use: "
            f"get_parsed_result(document_id='{doc_id}') to retrieve the text."
        ),
    })


async def token_upload_endpoint(request: Request):
    """One-time token upload: accepts file, parses it, returns document_id."""
    token = request.path_params.get("token", "")
    parser = request.query_params.get("parser", "mistral")

    if token not in _upload_tokens:
        return JSONResponse({"error": "Invalid or expired upload token"}, status_code=403)

    token_data = _upload_tokens.pop(token)

    # Check TTL
    if time.time() - token_data["created"] > UPLOAD_TOKEN_TTL:
        return JSONResponse({"error": "Upload token expired. Request a new one."}, status_code=403)

    if parser == "mistral":
        if not MISTRAL_API_KEY:
            return JSONResponse({"error": "No Mistral API key configured"}, status_code=500)
    elif not GOOGLE_DOCAI_CREDENTIALS_PATH:
        return JSONResponse({"error": "No credentials configured"}, status_code=500)

    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file")
        if not file:
            return JSONResponse({"error": "No file field in form"}, status_code=400)
        file_bytes = await file.read()
        filename = file.filename or token_data["filename"]
        mime_type = file.content_type or "application/pdf"
    else:
        file_bytes = await request.body()
        filename = request.headers.get("x-filename", token_data["filename"])
        mime_type = content_type if content_type and content_type != "application/octet-stream" else "application/pdf"

    if not file_bytes:
        return JSONResponse({"error": "Empty file"}, status_code=400)

    # Check cache first
    cached_id = _check_cache(file_bytes)
    if cached_id:
        entry = _parsed_results[cached_id]
        return JSONResponse({
            "document_id": cached_id,
            "filename": entry["filename"],
            "pages": entry["pages"],
            "chars": len(entry["text"]),
            "cached": True,
        })

    page_count = 0
    if mime_type == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            page_count = len(reader.pages)
        except Exception:
            page_count = 1
    else:
        page_count = 1

    try:
        if parser == "mistral":
            result = await _process_document_mistral(file_bytes, mime_type)
        else:
            result = await _process_document(file_bytes, mime_type)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    doc_id = str(uuid.uuid4())[:8]
    _store_result(doc_id, file_bytes, result, page_count, filename)

    return JSONResponse({
        "document_id": doc_id,
        "filename": filename,
        "pages": page_count,
        "chars": len(result),
        "parser": "mistral" if parser == "mistral" else "google",
    })


async def debug_processor_endpoint(request: Request):
    """Show processor info and available versions."""
    if not GOOGLE_DOCAI_CREDENTIALS_PATH:
        return JSONResponse({"error": "No credentials"}, status_code=500)
    try:
        client = _get_docai_client()
        name = client.processor_path(GCP_PROJECT_ID, GCP_LOCATION, "8a96e920607e3974")
        proc = await asyncio.to_thread(client.get_processor, name=name)
        versions = await asyncio.to_thread(client.list_processor_versions, parent=name)
        version_list = []
        for v in versions:
            version_list.append({
                "display_name": v.display_name,
                "id": v.name.split("/")[-1],
                "state": str(v.state),
                "model_type": str(v.model_type) if hasattr(v, 'model_type') else None,
            })
        return JSONResponse({
            "processor_name": proc.display_name,
            "type": proc.type_,
            "default_version": proc.default_processor_version,
            "state": str(proc.state),
            "versions": version_list,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_result_endpoint(request: Request):
    """Direct HTTP endpoint to retrieve parsed text (non-MCP access)."""
    doc_id = request.path_params.get("doc_id", "")
    if doc_id not in _parsed_results:
        return JSONResponse({"error": f"No document with id '{doc_id}'"}, status_code=404)
    return PlainTextResponse(_parsed_results[doc_id]["text"])


async def get_result_download_endpoint(request: Request):
    """Return parsed text as a direct file download with Content-Disposition header."""
    doc_id = request.path_params.get("doc_id", "")
    if doc_id not in _parsed_results:
        return JSONResponse({"error": f"No document with id '{doc_id}'"}, status_code=404)

    entry = _parsed_results[doc_id]
    text = entry["text"]
    original_name = entry.get("filename", "document")
    base_name = original_name.rsplit(".", 1)[0] if "." in original_name else original_name
    md_filename = f"{base_name}.md"

    return Response(
        content=text.encode("utf-8"),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{md_filename}"'},
    )


async def get_result_file_endpoint(request: Request):
    """Return parsed text as a downloadable file via openaiFileResponse for ChatGPT."""
    doc_id = request.path_params.get("doc_id", "")
    if doc_id not in _parsed_results:
        return JSONResponse({"error": f"No document with id '{doc_id}'"}, status_code=404)

    entry = _parsed_results[doc_id]
    text = entry["text"]
    original_name = entry.get("filename", "document")
    # Strip extension and add .md
    base_name = original_name.rsplit(".", 1)[0] if "." in original_name else original_name
    md_filename = f"{base_name}.md"

    content_b64 = base64.b64encode(text.encode("utf-8")).decode("utf-8")

    return JSONResponse({
        "openaiFileResponse": [
            {
                "name": md_filename,
                "mime_type": "text/markdown",
                "content": content_b64,
            }
        ]
    })


async def debug_parse_endpoint(request: Request):
    """Upload a PDF and return the RAW Document AI response structure for page 1."""
    if not GOOGLE_DOCAI_CREDENTIALS_PATH:
        return JSONResponse({"error": "No credentials configured"}, status_code=500)

    form = await request.form()
    upload = form.get("file")
    if not upload:
        return JSONResponse({"error": "No file"}, status_code=400)

    file_bytes = await upload.read()
    page_num = int(request.query_params.get("page", "1"))

    # Process just the requested page
    reader = PdfReader(io.BytesIO(file_bytes))
    if page_num > len(reader.pages):
        return JSONResponse({"error": f"Page {page_num} doesn't exist (doc has {len(reader.pages)} pages)"}, status_code=400)

    writer = PdfWriter()
    writer.add_page(reader.pages[page_num - 1])
    buf = io.BytesIO()
    writer.write(buf)
    single_page_bytes = buf.getvalue()

    client = _get_docai_client()
    resource_name = client.processor_path(GCP_PROJECT_ID, GCP_LOCATION, GCP_PROCESSOR_ID)
    raw_document = documentai.RawDocument(content=single_page_bytes, mime_type="application/pdf")
    process_options = documentai.ProcessOptions(
        layout_config=documentai.ProcessOptions.LayoutConfig(
            chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
                chunk_size=1000,
                include_ancestor_headings=True,
            ),
        ),
    )
    request_obj = documentai.ProcessRequest(
        name=resource_name,
        raw_document=raw_document,
        process_options=process_options,
    )
    result = await asyncio.to_thread(client.process_document, request=request_obj)
    doc = result.document

    # Build debug output
    page_data = []
    for i, page in enumerate(doc.pages):
        p = {
            "page_number": page.page_number,
            "width": page.dimension.width if page.dimension else None,
            "height": page.dimension.height if page.dimension else None,
            "detected_languages": [
                {"code": lang.language_code, "confidence": lang.confidence}
                for lang in (page.detected_languages or [])
            ],
            "paragraphs": len(page.paragraphs) if page.paragraphs else 0,
            "lines": len(page.lines) if page.lines else 0,
            "tokens": len(page.tokens) if page.tokens else 0,
            "tables": len(page.tables) if page.tables else 0,
            "form_fields": len(page.form_fields) if page.form_fields else 0,
            "visual_elements": len(page.visual_elements) if page.visual_elements else 0,
            "blocks": len(page.blocks) if page.blocks else 0,
        }

        # Sample table structures
        table_details = []
        for t, table in enumerate(page.tables or []):
            tbl = {
                "header_rows": len(table.header_rows),
                "body_rows": len(table.body_rows),
                "header_cells": [],
                "first_body_row_cells": [],
            }
            for row in table.header_rows[:1]:
                for cell in row.cells:
                    tbl["header_cells"].append(_get_text_from_layout(cell.layout, doc.text).strip()[:50])
            for row in table.body_rows[:1]:
                for cell in row.cells:
                    tbl["first_body_row_cells"].append(_get_text_from_layout(cell.layout, doc.text).strip()[:50])
            table_details.append(tbl)
        p["table_details"] = table_details

        page_data.append(p)

    # Chunks from layout parser — dump all available fields
    chunks = []
    for chunk in (doc.chunked_document.chunks if doc.chunked_document else []):
        c = {
            "id": chunk.chunk_id,
            "content_preview": (chunk.content or "")[:200],
            "content_len": len(chunk.content) if chunk.content else 0,
            "all_fields": [f for f in dir(chunk) if not f.startswith('_')],
        }
        # Try all possible page-related fields
        for attr in ['page_span', 'page_headers', 'page_footers', 'source_block_ids']:
            val = getattr(chunk, attr, None)
            if val:
                try:
                    items = list(val)
                    c[attr] = [str(item)[:200] for item in items[:3]]
                except Exception as e:
                    c[attr] = f"error: {e}"
            else:
                c[attr] = None
        chunks.append(c)

    return JSONResponse({
        "requested_page": page_num,
        "total_document_text_chars": len(doc.text),
        "document_text_preview": doc.text[:2000],
        "pages": page_data,
        "chunks": chunks[:20],
        "entities_count": len(doc.entities) if doc.entities else 0,
    })


async def parse_chatgpt_endpoint(request: Request):
    """Parse a document uploaded via ChatGPT Actions using openaiFileIdRefs.
    ChatGPT sends: {"openaiFileIdRefs": [{"name": "...", "download_link": "...", "mime_type": "..."}]}"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    parser = body.get("parser", "mistral")
    file_refs = body.get("openaiFileIdRefs", [])

    if not file_refs:
        return JSONResponse({"error": "No files provided. Upload a PDF and try again."}, status_code=400)

    ref = file_refs[0]
    # openaiFileIdRefs items are objects with name, id, mime_type, download_link
    if isinstance(ref, dict):
        download_url = ref.get("download_link") or ref.get("url")
        filename = ref.get("name", "document.pdf")
        mime_type = ref.get("mime_type", "application/pdf")
    elif isinstance(ref, str):
        download_url = ref
        filename = "document.pdf"
        mime_type = "application/pdf"
    else:
        return JSONResponse({"error": f"Unexpected file reference format: {type(ref)}"}, status_code=400)

    if not download_url:
        return JSONResponse({
            "error": "No download link in file reference. File reference received: " + json.dumps(ref, default=str)[:500]
        }, status_code=400)

    if parser == "mistral" and not MISTRAL_API_KEY:
        return JSONResponse({"error": "No Mistral API key configured"}, status_code=500)
    if parser == "google" and not GOOGLE_DOCAI_CREDENTIALS_PATH:
        return JSONResponse({"error": "No Google credentials configured"}, status_code=500)

    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as dl_client:
            resp = await dl_client.get(download_url)
            resp.raise_for_status()
            file_bytes = resp.content
    except Exception as e:
        return JSONResponse({"error": f"Failed to download file from ChatGPT: {str(e)}"}, status_code=400)

    if not file_bytes:
        return JSONResponse({"error": "Downloaded file was empty"}, status_code=400)

    # Cache check
    cached_id = _check_cache(file_bytes)
    if cached_id:
        entry = _parsed_results[cached_id]
        return JSONResponse({
            "document_id": cached_id,
            "filename": entry["filename"],
            "pages": entry["pages"],
            "chars": len(entry["text"]),
            "parser": parser,
            "cached": True,
        })

    # Count pages
    page_count = 0
    if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            page_count = len(reader.pages)
        except Exception:
            page_count = 1
    else:
        page_count = 1

    try:
        if parser == "mistral":
            result = await _process_document_mistral(file_bytes, mime_type)
        else:
            result = await _process_document(file_bytes, mime_type)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    doc_id = str(uuid.uuid4())[:8]
    _store_result(doc_id, file_bytes, result, page_count, filename)

    return JSONResponse({
        "document_id": doc_id,
        "filename": filename,
        "pages": page_count,
        "chars": len(result),
        "parser": parser,
    })


async def parse_url_endpoint(request: Request):
    """Parse a document from a URL. JSON body: {"url": "...", "parser": "mistral"|"google"}
    Designed for ChatGPT Custom GPT Actions."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body. Expected: {\"url\": \"...\", \"parser\": \"mistral\"}"}, status_code=400)

    url = body.get("url")
    if not url:
        return JSONResponse({"error": "Missing 'url' field"}, status_code=400)

    parser = body.get("parser", "mistral")

    if parser == "mistral":
        if not MISTRAL_API_KEY:
            return JSONResponse({"error": "No Mistral API key configured"}, status_code=500)
    elif not GOOGLE_DOCAI_CREDENTIALS_PATH:
        return JSONResponse({"error": "No Google credentials configured"}, status_code=500)

    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            file_bytes = resp.content
    except Exception as e:
        return JSONResponse({"error": f"Failed to download: {str(e)}"}, status_code=400)

    # Auto-detect mime type
    lower_url = url.lower().split("?")[0]
    if lower_url.endswith(".png"):
        mime_type = "image/png"
    elif lower_url.endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif lower_url.endswith(".tiff"):
        mime_type = "image/tiff"
    else:
        mime_type = "application/pdf"

    filename = url.split("/")[-1].split("?")[0] or "document.pdf"

    # Cache check
    cached_id = _check_cache(file_bytes)
    if cached_id:
        entry = _parsed_results[cached_id]
        return JSONResponse({
            "document_id": cached_id,
            "filename": entry["filename"],
            "pages": entry["pages"],
            "chars": len(entry["text"]),
            "parser": parser,
            "cached": True,
        })

    # Count pages
    page_count = 0
    if mime_type == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            page_count = len(reader.pages)
        except Exception:
            page_count = 1
    else:
        page_count = 1

    try:
        if parser == "mistral":
            result = await _process_document_mistral(file_bytes, mime_type)
        else:
            result = await _process_document(file_bytes, mime_type)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    doc_id = str(uuid.uuid4())[:8]
    _store_result(doc_id, file_bytes, result, page_count, filename)

    return JSONResponse({
        "document_id": doc_id,
        "filename": filename,
        "pages": page_count,
        "chars": len(result),
        "parser": parser,
    })


async def widget_endpoint(request):
    """Serve the ChatGPT App widget HTML."""
    widget_path = os.path.join(os.path.dirname(__file__), "public", "parser-widget.html")
    with open(widget_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content, media_type="text/html;profile=mcp-app")


from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


# --- Combined app: ChatGPT streamable HTTP as base, Claude SSE + HTTP routes injected ---

# ChatGPT MCP app (streamable HTTP with /mcp endpoint)
_chatgpt_app = mcp_chatgpt.streamable_http_app()

# Inject Claude SSE + all HTTP routes before the /mcp route
_custom_routes = [
    Route("/", homepage),
    Route("/favicon-32.png", favicon),
    Route("/favicon.ico", favicon),
    Route("/health", health),
    Route("/parse", parse_endpoint, methods=["POST"]),
    Route("/parse-url", parse_url_endpoint, methods=["POST"]),
    Route("/parse-chatgpt", parse_chatgpt_endpoint, methods=["POST"]),
    Route("/debug-parse", debug_parse_endpoint, methods=["POST"]),
    Route("/debug-processor", debug_processor_endpoint),
    Route("/upload/{token}", token_upload_endpoint, methods=["POST"]),
    Route("/result/{doc_id}", get_result_endpoint),
    Route("/result-file/{doc_id}", get_result_file_endpoint),
    Route("/download/{doc_id}", get_result_download_endpoint),
    Route("/widget/parser", widget_endpoint),
    Route("/sse", endpoint=handle_sse),
    Mount("/messages/", app=sse.handle_post_message),
]
_chatgpt_app.router.routes = _custom_routes + list(_chatgpt_app.router.routes)

# Add CORS middleware
_chatgpt_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app = _chatgpt_app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
