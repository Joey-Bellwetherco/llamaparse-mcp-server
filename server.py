import os
import io
import json
import uuid
import time
import base64
import asyncio
import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, HTMLResponse
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


def _process_single_chunk(file_content: bytes, mime_type: str) -> str:
    """Send a single document chunk to Document AI and return parsed text with tables."""
    client = _get_docai_client()
    resource_name = client.processor_path(GCP_PROJECT_ID, GCP_LOCATION, GCP_PROCESSOR_ID)
    raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)

    request = documentai.ProcessRequest(
        name=resource_name,
        raw_document=raw_document,
    )
    result = client.process_document(request=request)
    document = result.document

    output_parts = [document.text]

    for i, page in enumerate(document.pages):
        tables = _extract_tables_from_page(page, document.text)
        if tables:
            page_num = page.page_number if page.page_number else i + 1
            for j, table_md in enumerate(tables):
                output_parts.append(f"\n[Table {j+1} on page {page_num}]\n{table_md}")

    return "\n\n".join(output_parts)


async def _process_document(file_content: bytes, mime_type: str = "application/pdf") -> str:
    """Process a document, splitting large PDFs into parallel chunks."""
    if mime_type == "application/pdf":
        chunks = _split_pdf(file_content)
    else:
        chunks = [file_content]

    if len(chunks) == 1:
        return await asyncio.to_thread(_process_single_chunk, chunks[0], mime_type)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def process_with_limit(chunk: bytes) -> str:
        async with semaphore:
            return await asyncio.to_thread(_process_single_chunk, chunk, mime_type)

    results = await asyncio.gather(*[process_with_limit(c) for c in chunks])
    return "\n\n".join(results)


# --- Parsed result cache + upload tokens ---
_parsed_results: dict[str, dict] = {}  # id -> {"text": str, "pages": int, "filename": str}
_upload_tokens: dict[str, dict] = {}   # token -> {"created": float, "filename": str}
UPLOAD_TOKEN_TTL = 300  # 5 minutes

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

    # Split by page markers and return requested range
    # Document AI text doesn't have explicit page markers, so we split roughly
    lines = full_text.split("\n")
    total_lines = len(lines)
    if total_pages > 0:
        lines_per_page = max(1, total_lines // total_pages)
    else:
        lines_per_page = total_lines

    start_line = (page_start - 1) * lines_per_page
    end_line = (page_end * lines_per_page) if page_end > 0 else total_lines

    selected = "\n".join(lines[start_line:end_line])
    return (
        f"[Document: {entry['filename']}, showing pages {page_start}-"
        f"{page_end if page_end > 0 else total_pages} of {total_pages}]\n\n{selected}"
    )


@mcp.tool()
async def list_parsed_documents() -> str:
    """List all documents that have been parsed and are available for retrieval.

    Returns document IDs, filenames, and page counts.
    """
    if not _parsed_results:
        return f"No parsed documents available. Upload a file at {UPLOAD_URL} to get started."

    lines = []
    for doc_id, entry in _parsed_results.items():
        lines.append(
            f"- {entry['filename']}: {entry['pages']} pages, "
            f"{len(entry['text'])} chars (id: {doc_id})"
        )
    return "Parsed documents:\n" + "\n".join(lines)


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
        _parsed_results[doc_id] = {
            "text": result,
            "pages": page_count,
            "filename": filename,
        }

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
            f"Cached documents: {len(_parsed_results)}"
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
    _parsed_results[doc_id] = {
        "text": result,
        "pages": page_count,
        "filename": filename,
    }

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
    _parsed_results[doc_id] = {
        "text": result,
        "pages": page_count,
        "filename": filename,
    }

    return JSONResponse({
        "document_id": doc_id,
        "filename": filename,
        "pages": page_count,
        "chars": len(result),
    })


async def get_result_endpoint(request: Request):
    """Direct HTTP endpoint to retrieve parsed text (non-MCP access)."""
    doc_id = request.path_params.get("doc_id", "")
    if doc_id not in _parsed_results:
        return JSONResponse({"error": f"No document with id '{doc_id}'"}, status_code=404)
    return PlainTextResponse(_parsed_results[doc_id]["text"])


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
        Route("/health", health),
        Route("/parse", parse_endpoint, methods=["POST"]),
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
