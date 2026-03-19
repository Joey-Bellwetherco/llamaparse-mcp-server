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
</style>
</head>
<body>
<div class="container">
  <h1>Document AI Parser</h1>
  <p class="sub">Upload a PDF or image to parse with Google Document AI OCR</p>
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

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) uploadFile(fileInput.files[0]); });

async function uploadFile(file) {
  status.innerHTML = '<p class="parsing">Parsing ' + file.name + '... (this may take a minute for large files)</p>';
  const form = new FormData();
  form.append('file', file);
  try {
    const resp = await fetch('/parse', { method: 'POST', body: form });
    const data = await resp.json();
    if (data.error) { status.innerHTML = '<p class="error">Error: ' + data.error + '</p>'; return; }
    status.innerHTML = `
      <div class="success">
        <p>Parsed <strong>${data.filename}</strong> (${data.pages} pages, ${data.chars.toLocaleString()} chars)</p>
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


CHUNK_SIZE = 15  # Document AI online limit is 15 pages per request
MAX_CONCURRENT = 10  # Max parallel requests to Document AI


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

mcp = FastMCP("Document AI MCP")


@mcp.tool()
async def get_parsed_result(
    document_id: str,
    page_start: int = 1,
    page_end: int = 0,
) -> str:
    """Retrieve parsed text from a document that was uploaded and parsed via POST /parse.

    The document_id is returned by the /parse endpoint. You can retrieve the full text
    or a page range to avoid overloading the context window.

    Args:
        document_id: The ID returned from POST /parse
        page_start: First page to return (1-indexed, default: 1)
        page_end: Last page to return (0 = all remaining pages)
    """
    if document_id not in _parsed_results:
        return (
            f"Error: No parsed document with id '{document_id}'. "
            "Upload and parse a file first via POST /parse endpoint."
        )

    entry = _parsed_results[document_id]
    full_text = entry["text"]
    total_pages = entry["pages"]

    # If no page range requested, return full text
    if page_start <= 1 and page_end <= 0:
        return f"[Document: {entry['filename']}, {total_pages} pages]\n\n{full_text}"

    # Split by [Page N] markers for accurate page-range extraction
    page_sections = re.split(r'(?=\[Page \d+\])', full_text)
    page_sections = [s for s in page_sections if s.strip()]

    # Build a dict of page_num -> text
    page_map = {}
    for section in page_sections:
        match = re.match(r'\[Page (\d+)\]', section)
        if match:
            pnum = int(match.group(1))
            page_map[pnum] = section

    end = page_end if page_end > 0 else total_pages
    selected_parts = []
    for p in range(page_start, end + 1):
        if p in page_map:
            selected_parts.append(page_map[p])

    selected = "\n\n".join(selected_parts) if selected_parts else "(No content for requested pages)"
    return (
        f"[Document: {entry['filename']}, showing pages {page_start}-"
        f"{end} of {total_pages}]\n\n{selected}"
    )


@mcp.tool()
async def get_upload_url(
    filename: str = "document.pdf",
) -> str:
    """IMPORTANT: When a user asks to parse a PDF, ALWAYS use this tool first.
    DO NOT base64-encode the file. DO NOT try to send file contents as text.

    This returns a one-time upload URL. After calling this tool:
    1. Find the PDF file path in the sandbox (user attachments are at /uploads/ or similar)
    2. Run the curl command in code execution to upload the file to the URL
    3. Parse the JSON response to get the document_id
    4. Call get_parsed_result(document_id) to retrieve the parsed text

    Args:
        filename: Name of the file being uploaded
    """
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
        f"NEXT STEPS — you MUST do these in code execution:\n\n"
        f"Step 1: Find the file. Run: ls /uploads/ /tmp/ ~/ to locate the PDF.\n\n"
        f"Step 2: Upload it with curl:\n"
        f'curl -s -X POST "{upload_url}" -F "file=@FILE_PATH"\n\n'
        f"Step 3: The response JSON contains a document_id. "
        f"Call get_parsed_result(document_id) to retrieve the parsed text.\n\n"
        f"DO NOT try to base64-encode or read the file contents into the conversation."
    )


@mcp.tool()
async def parse_document_from_url(
    url: str,
    mime_type: str = "application/pdf",
) -> str:
    """Parse a document from a URL using Google Document AI with visual OCR.

    Provide a URL to a document (PDF or image) and get back the parsed text content.
    For large documents, the result is cached — use get_parsed_result to retrieve pages.

    Args:
        url: Direct URL to a document file
        mime_type: MIME type of the document (default: application/pdf)
    """
    if not GOOGLE_DOCAI_CREDENTIALS_PATH:
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
                f"[Cached: {entry['filename']}, {page_count} pages, {len(result)} chars]\n"
                f"Result is large. Use get_parsed_result(document_id='{cached_id}') to retrieve "
                f"page ranges. Example: get_parsed_result('{cached_id}', page_start=1, page_end=10)"
            )

    try:
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

        # Cache the result
        doc_id = str(uuid.uuid4())[:8]
        _store_result(doc_id, file_bytes, result, page_count, filename)

        # For small results return inline, for large ones return summary + ID
        if len(result) < 50000:
            return f"[Parsed {filename}, {page_count} pages, id: {doc_id}]\n\n{result}"
        else:
            return (
                f"[Parsed {filename}, {page_count} pages, {len(result)} chars]\n"
                f"Result is large. Use get_parsed_result(document_id='{doc_id}') to retrieve "
                f"page ranges. Example: get_parsed_result('{doc_id}', page_start=1, page_end=10)"
            )
    except Exception as e:
        return f"Error: {str(e)}"


UPLOAD_URL = os.environ.get("PUBLIC_URL", "https://bw-parse-mcp-server.up.railway.app")


@mcp.tool()
async def check_status() -> str:
    """Check if the Document AI MCP server is configured and ready.

    Also returns the upload URL where users can upload PDFs for parsing.
    """
    if GOOGLE_DOCAI_CREDENTIALS_PATH:
        return (
            f"Document AI MCP is running and ready.\n\n"
            f"IMPORTANT: To parse a PDF, NEVER base64-encode it. Instead:\n"
            f"1. Call get_upload_url(filename) to get a one-time upload URL\n"
            f"2. In code execution, run: curl -s -X POST \"<url>\" -F \"file=@/path/to/file.pdf\"\n"
            f"3. Call get_parsed_result(document_id) with the ID from the curl response\n\n"
            f"For URLs: call parse_document_from_url(url) directly.\n"
            f"Manual upload: {UPLOAD_URL}\n\n"
            f"Processor: Layout Parser with Gemini"
        )
    else:
        return "Document AI MCP is running but no credentials are configured. Set GOOGLE_DOCAI_CREDENTIALS_PATH."


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
    """Simple drag-and-drop upload page."""
    return HTMLResponse(UPLOAD_PAGE_HTML)


async def favicon(request):
    """Serve the favicon."""
    favicon_path = os.path.join(os.path.dirname(__file__), "favicon-32.png")
    with open(favicon_path, "rb") as f:
        return Response(f.read(), media_type="image/png")


async def health(request):
    return JSONResponse({"status": "ok"})


async def parse_endpoint(request: Request):
    """Upload + parse in one step. Returns a document_id for MCP tool retrieval."""
    if not GOOGLE_DOCAI_CREDENTIALS_PATH:
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
        "message": (
            f"Parsed successfully. In Claude, use: "
            f"get_parsed_result(document_id='{doc_id}') to retrieve the text."
        ),
    })


async def token_upload_endpoint(request: Request):
    """One-time token upload: accepts file, parses it, returns document_id."""
    token = request.path_params.get("token", "")

    if token not in _upload_tokens:
        return JSONResponse({"error": "Invalid or expired upload token"}, status_code=403)

    token_data = _upload_tokens.pop(token)

    # Check TTL
    if time.time() - token_data["created"] > UPLOAD_TOKEN_TTL:
        return JSONResponse({"error": "Upload token expired. Request a new one."}, status_code=403)

    if not GOOGLE_DOCAI_CREDENTIALS_PATH:
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
    })


async def debug_processor_endpoint(request: Request):
    """Show processor info and available versions."""
    if not GOOGLE_DOCAI_CREDENTIALS_PATH:
        return JSONResponse({"error": "No credentials"}, status_code=500)
    try:
        client = _get_docai_client()
        name = client.processor_path(GCP_PROJECT_ID, GCP_LOCATION, GCP_PROCESSOR_ID)
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

    # Chunks from layout parser
    chunks = []
    for chunk in (doc.chunked_document.chunks if doc.chunked_document else []):
        chunks.append({
            "id": chunk.chunk_id,
            "content": chunk.content[:500] if chunk.content else "",
            "page_span": [
                {"page": ps.page_start, "end": ps.page_end}
                for ps in (chunk.page_span or [])
            ],
        })

    return JSONResponse({
        "requested_page": page_num,
        "total_document_text_chars": len(doc.text),
        "document_text_preview": doc.text[:2000],
        "pages": page_data,
        "chunks": chunks[:20],
        "entities_count": len(doc.entities) if doc.entities else 0,
    })


async def oauth_protected_resource(request):
    return JSONResponse({
        "resource": f"https://{request.headers.get('host', 'localhost')}",
        "bearer_methods_supported": [],
        "resource_documentation": "https://github.com/Joey-Bellwetherco/llamaparse-mcp-server",
    })


async def oauth_authorization_server(request):
    return JSONResponse({
        "issuer": f"https://{request.headers.get('host', 'localhost')}",
        "authorization_endpoint": f"https://{request.headers.get('host', 'localhost')}/authorize",
        "token_endpoint": f"https://{request.headers.get('host', 'localhost')}/token",
        "response_types_supported": ["code"],
    })


async def register(request):
    body = await request.json()
    return JSONResponse({
        "client_id": "docai-public-client",
        "client_secret": "",
        "client_name": body.get("client_name", "Claude"),
        "redirect_uris": body.get("redirect_uris", []),
    })


app = Starlette(
    routes=[
        Route("/", homepage),
        Route("/favicon-32.png", favicon),
        Route("/favicon.ico", favicon),
        Route("/health", health),
        Route("/parse", parse_endpoint, methods=["POST"]),
        Route("/debug-parse", debug_parse_endpoint, methods=["POST"]),
        Route("/debug-processor", debug_processor_endpoint),
        Route("/upload/{token}", token_upload_endpoint, methods=["POST"]),
        Route("/result/{doc_id}", get_result_endpoint),
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
        Route("/.well-known/oauth-protected-resource", oauth_protected_resource),
        Route("/.well-known/oauth-authorization-server", oauth_authorization_server),
        Route("/register", register, methods=["POST"]),
    ],
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
