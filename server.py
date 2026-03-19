import os
import io
import re
import json
import uuid
import time
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
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from pypdf import PdfReader, PdfWriter
import pymupdf

load_dotenv()

UPLOAD_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" type="image/png" href="/favicon-32.png">
<title>Databricks Document Parser</title>
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
  <h1>Databricks Document Parser</h1>
  <p class="sub">Upload a PDF, image, or Office document to parse with Databricks AI</p>
  <div class="drop-zone" id="dropZone">
    <div class="icon">📄</div>
    <p>Drag & drop a file here, or click to browse</p>
    <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg,.docx,.pptx">
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

# Databricks config
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")  # PAT (legacy)
DATABRICKS_CLIENT_ID = os.environ.get("DATABRICKS_CLIENT_ID", "")  # OAuth SP
DATABRICKS_CLIENT_SECRET = os.environ.get("DATABRICKS_CLIENT_SECRET", "")  # OAuth SP
DATABRICKS_WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
DATABRICKS_VOLUME_PATH = os.environ.get(
    "DATABRICKS_VOLUME_PATH",
    "/Volumes/bellwether_dev/default/staging/tmp"
)

# Auth check — need either PAT or OAuth SP credentials
_has_databricks_creds = bool(DATABRICKS_HOST and (
    DATABRICKS_TOKEN or (DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET)
))

_workspace_client = None
_warehouse_id = None


def _get_databricks_client() -> WorkspaceClient:
    """Create or return cached Databricks WorkspaceClient.

    Supports both OAuth service principal (preferred) and PAT (legacy).
    The SDK auto-detects auth method from the provided credentials.
    """
    global _workspace_client
    if _workspace_client is None:
        kwargs = {"host": DATABRICKS_HOST}
        if DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET:
            kwargs["client_id"] = DATABRICKS_CLIENT_ID
            kwargs["client_secret"] = DATABRICKS_CLIENT_SECRET
        elif DATABRICKS_TOKEN:
            kwargs["token"] = DATABRICKS_TOKEN
        _workspace_client = WorkspaceClient(**kwargs)
    return _workspace_client


def _get_warehouse_id() -> str:
    """Return configured warehouse ID or auto-discover one."""
    global _warehouse_id
    if _warehouse_id is None:
        if DATABRICKS_WAREHOUSE_ID:
            _warehouse_id = DATABRICKS_WAREHOUSE_ID
        else:
            w = _get_databricks_client()
            for wh in w.warehouses.list():
                if wh.enable_serverless_compute or str(wh.state) == "RUNNING":
                    _warehouse_id = wh.id
                    break
            if not _warehouse_id:
                raise RuntimeError("No available SQL warehouse found. Set DATABRICKS_WAREHOUSE_ID.")
    return _warehouse_id


CHUNK_SIZE = 1  # Pages per parallel chunk (1 = max parallelism)
MAX_CONCURRENT = 15  # Max parallel SQL statements


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


def _reconstruct_page_text(rows: list, page_offset: int = 0) -> str:
    """Convert SQL result rows into structured text with [Page N] markers.

    rows: list of [page_idx, elem_idx, element_type, content, description]
    page_offset: added to page_idx for correct numbering in chunked processing
    """
    if not rows:
        return ""

    pages: dict[int, list] = {}
    for row in rows:
        page_idx = int(row[0]) if row[0] is not None else 0
        elem_type = (row[2] or "text").strip()
        content = (row[3] or "").strip()
        description = (row[4] or "").strip() if len(row) > 4 else ""

        # Use content if available, fall back to description (e.g. for figures)
        text = content or description
        if not text:
            continue

        if page_idx not in pages:
            pages[page_idx] = []

        if elem_type == "table":
            pages[page_idx].append(f"[Table]\n{text}")
        elif elem_type == "figure":
            if content and description:
                pages[page_idx].append(f"[Figure: {description}]\n{content}")
            elif description:
                pages[page_idx].append(f"[Figure: {description}]")
            else:
                pages[page_idx].append(content)
        elif elem_type == "title":
            pages[page_idx].append(f"# {text}")
        elif elem_type == "section_header":
            pages[page_idx].append(f"## {text}")
        else:
            pages[page_idx].append(text)

    output_parts = []
    for page_idx in sorted(pages.keys()):
        page_num = page_idx + 1 + page_offset
        output_parts.append(f"[Page {page_num}]")
        output_parts.append("\n".join(pages[page_idx]))

    return "\n\n".join(output_parts)


_PARSE_SQL_TEMPLATE = """
WITH parsed AS (
    SELECT ai_parse_document(content, map(
        'version', '2.0',
        'descriptionElementTypes', '*'
    )) AS doc
    FROM read_files('{volume_path}', format => 'binaryFile')
),
elements AS (
    SELECT posexplode(
        variant_get(doc, '$.document.elements', 'ARRAY<VARIANT>')
    ) AS (elem_idx, element)
    FROM parsed
)
SELECT
    variant_get(element, '$.bbox[0].page_id', 'INT') AS page_idx,
    elem_idx,
    variant_get(element, '$.type', 'STRING') AS element_type,
    variant_get(element, '$.content', 'STRING') AS content,
    variant_get(element, '$.description', 'STRING') AS description
FROM elements
ORDER BY page_idx, elem_idx
"""


async def _parse_single_chunk(
    w: WorkspaceClient,
    warehouse_id: str,
    chunk_bytes: bytes,
    page_offset: int,
    semaphore: asyncio.Semaphore,
) -> str:
    """Upload one chunk to Volume, parse via SQL, return text with offset page numbers."""
    temp_filename = f"{uuid.uuid4().hex[:12]}.pdf"
    volume_path = f"{DATABRICKS_VOLUME_PATH}/{temp_filename}"

    async with semaphore:
        try:
            await asyncio.to_thread(
                w.files.upload,
                file_path=volume_path,
                contents=io.BytesIO(chunk_bytes),
                overwrite=True,
            )

            sql = _PARSE_SQL_TEMPLATE.format(volume_path=volume_path)
            response = await asyncio.to_thread(
                w.statement_execution.execute_statement,
                warehouse_id=warehouse_id,
                statement=sql,
                wait_timeout="50s",
            )

            # Poll if still running (warehouse cold-start or large chunk)
            while response.status.state in (StatementState.PENDING, StatementState.RUNNING):
                await asyncio.sleep(2)
                response = await asyncio.to_thread(
                    w.statement_execution.get_statement,
                    statement_id=response.statement_id,
                )

            if response.status.state != StatementState.SUCCEEDED:
                error_msg = response.status.error.message if response.status.error else "Unknown SQL error"
                return f"[Page {page_offset + 1}]\n(Error parsing chunk: {error_msg})"

            return _reconstruct_page_text(response.result.data_array or [], page_offset)

        finally:
            try:
                await asyncio.to_thread(w.files.delete, file_path=volume_path)
            except Exception:
                pass


def _extract_pymupdf_reference(pdf_bytes: bytes) -> dict[int, str]:
    """Extract per-page native text from PDF using PyMuPDF as a reference.

    Returns dict of {page_num (1-indexed): raw_text}.
    Only returns pages that have meaningful native text (>20 chars).
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    reference = {}
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if len(text) > 20:
            reference[i + 1] = text
    doc.close()
    return reference


_PROOFREAD_SQL_TEMPLATE = """
SELECT ai_query(
    'databricks-claude-sonnet-4-6',
    'You are a precision document correction engine. You have two inputs:

PARSED OUTPUT — from an AI document parser. It has good structure and layout (headings, tables, figures) but may contain small errors from OCR/visual parsing.

REFERENCE TEXT — extracted directly from the native PDF text layer. This is the GROUND TRUTH for any text that appears in both. It is character-perfect but has no structure, may contain hidden/overlapping text artifacts, and has no table formatting.

YOUR JOB: Use the reference text to correct errors in the parsed output while keeping the parsed output''s structure intact.

CORRECTIONS TO MAKE (use reference as ground truth):
- Wrong characters: "0" vs "O", "1" vs "l" vs "I", "rn" vs "m", "cl" vs "d", "S" vs "$", etc.
- Wrong or slightly off numbers, dollar amounts, percentages, dates, rates
- Misread decimals: "7.6" vs "76", commas vs periods in numbers
- Merged words or broken words (missing/extra spaces)
- Swapped, shifted, or misaligned table columns or rows
- Missing text that exists in the reference but was dropped from parsed output
- Truncated words or sentences
- Wrong currency symbols, units, or special characters (%, ±, °, ™, etc.)
- Scrambled or reordered text within a line
- Headers/footers/page numbers bleeding into body content
- Footnote markers mixed into running text
- Repeated or duplicated text that only appears once in reference
- Wrong capitalization or case changes
- Accented characters or unicode that got replaced with ASCII equivalents

KEEP UNCHANGED:
- All [Page N], [Table], [Figure:], ##, # markers and structure
- Layout and ordering of sections
- Any text that matches between both sources

DO NOT:
- Add text that appears ONLY in the reference but not in the parsed output (reference may contain hidden layers, watermarks, or overlapping objects)
- Add commentary, notes, or explanations
- Reformat tables or change markdown structure
- Invent or hallucinate content

If the parsed output and reference agree, return the parsed output unchanged.
Return ONLY the corrected text. Nothing else.

PARSED OUTPUT:
{parsed_text}

REFERENCE TEXT:
{reference_text}',
    failOnError => false
).response
"""


async def _proofread_page(
    w: WorkspaceClient,
    warehouse_id: str,
    page_num: int,
    parsed_text: str,
    reference_text: str,
    semaphore: asyncio.Semaphore,
) -> tuple[int, str]:
    """Use ai_query to proofread a single page against PyMuPDF reference."""
    async with semaphore:
        # Escape single quotes for SQL, keep full text (model handles up to ~16k tokens)
        parsed_trunc = parsed_text[:12000].replace("'", "''")
        ref_trunc = reference_text[:12000].replace("'", "''")

        sql = _PROOFREAD_SQL_TEMPLATE.format(
            parsed_text=parsed_trunc,
            reference_text=ref_trunc,
        )

        response = await asyncio.to_thread(
            w.statement_execution.execute_statement,
            warehouse_id=warehouse_id,
            statement=sql,
            wait_timeout="50s",
        )

        while response.status.state in (StatementState.PENDING, StatementState.RUNNING):
            await asyncio.sleep(2)
            response = await asyncio.to_thread(
                w.statement_execution.get_statement,
                statement_id=response.statement_id,
            )

        if (response.status.state == StatementState.SUCCEEDED
                and response.result.data_array
                and response.result.data_array[0][0]):
            return page_num, response.result.data_array[0][0]

        # If proofreading fails, return original parsed text
        return page_num, parsed_text


async def _process_document(file_content: bytes, mime_type: str = "application/pdf") -> str:
    """Process a document with Databricks AI, proofread against PyMuPDF reference.

    1. Databricks ai_parse_document — primary parser (sees visual layout)
    2. PyMuPDF — extracts native text as ground truth reference
    3. ai_query — proofreads parsed output against reference, fixes character
       errors, wrong values, and misaligned columns
    """
    # Step 1: Always parse with Databricks (primary)
    databricks_text = await _process_document_databricks(file_content, mime_type)

    # Step 2: If PDF, extract PyMuPDF reference and proofread
    if mime_type != "application/pdf":
        return databricks_text

    reference = await asyncio.to_thread(_extract_pymupdf_reference, file_content)
    if not reference:
        # No native text to proofread against (fully scanned doc)
        return databricks_text

    # Split Databricks output into per-page sections
    page_sections = re.split(r'(?=\[Page \d+\])', databricks_text)
    page_sections = [s for s in page_sections if s.strip()]

    page_map = {}
    for section in page_sections:
        match = re.match(r'\[Page (\d+)\]', section)
        if match:
            page_map[int(match.group(1))] = section

    # Step 3: Proofread pages that have both parsed and reference text
    w = _get_databricks_client()
    warehouse_id = _get_warehouse_id()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    proofread_tasks = []
    for page_num, parsed_section in page_map.items():
        if page_num in reference:
            proofread_tasks.append(
                _proofread_page(w, warehouse_id, page_num, parsed_section, reference[page_num], semaphore)
            )

    if proofread_tasks:
        results = await asyncio.gather(*proofread_tasks)
        for page_num, corrected_text in results:
            page_map[page_num] = corrected_text

    combined = "\n\n".join(page_map[p] for p in sorted(page_map.keys()))
    return combined or "(No content extracted from document)"


async def _process_document_databricks(file_content: bytes, mime_type: str = "application/pdf") -> str:
    """Process a document via Databricks ai_parse_document with parallel chunking."""
    w = _get_databricks_client()
    warehouse_id = _get_warehouse_id()

    if mime_type == "application/pdf":
        chunks = _split_pdf(file_content)
    else:
        chunks = [file_content]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        _parse_single_chunk(w, warehouse_id, chunk, i * CHUNK_SIZE, semaphore)
        for i, chunk in enumerate(chunks)
    ]
    results = await asyncio.gather(*tasks)
    combined = "\n\n".join(r for r in results if r)
    return combined or "(No content extracted from document)"


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

mcp = FastMCP("Databricks Document Parser MCP")


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
    """Parse a document from a URL using Databricks AI.

    Provide a URL to a document (PDF, image, or Office document) and get back the parsed text content.
    For large documents, the result is cached — use get_parsed_result to retrieve pages.

    Args:
        url: Direct URL to a document file
        mime_type: MIME type of the document (default: application/pdf)
    """
    if not _has_databricks_creds:
        return "Error: No Databricks credentials configured. Set DATABRICKS_HOST + DATABRICKS_CLIENT_ID/CLIENT_SECRET (OAuth) or DATABRICKS_TOKEN (PAT)."

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
    elif lower_url.endswith(".docx"):
        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif lower_url.endswith(".pptx"):
        mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

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
    """Check if the Databricks Document Parser MCP server is configured and ready."""
    if _has_databricks_creds:
        warehouse_info = ""
        try:
            wid = _get_warehouse_id()
            warehouse_info = f"\nSQL Warehouse: {wid}"
        except Exception as e:
            warehouse_info = f"\nWarehouse discovery: {e}"

        return (
            f"Databricks Document Parser MCP is running and ready.\n\n"
            f"IMPORTANT: To parse a PDF, NEVER base64-encode it. Instead:\n"
            f"1. Call get_upload_url(filename) to get a one-time upload URL\n"
            f"2. In code execution, run: curl -s -X POST \"<url>\" -F \"file=@/path/to/file.pdf\"\n"
            f"3. Call get_parsed_result(document_id) with the ID from the curl response\n\n"
            f"For URLs: call parse_document_from_url(url) directly.\n"
            f"Manual upload: {UPLOAD_URL}\n\n"
            f"Host: {DATABRICKS_HOST}{warehouse_info}\n"
            f"Volume: {DATABRICKS_VOLUME_PATH}\n"
            f"Supported formats: PDF, PNG, JPG, DOCX, PPTX"
        )
    else:
        return "Databricks Document Parser MCP is running but no credentials are configured. Set DATABRICKS_HOST + DATABRICKS_CLIENT_ID/CLIENT_SECRET (OAuth) or DATABRICKS_TOKEN (PAT)."


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
    status = {"status": "ok"}
    if _has_databricks_creds:
        try:
            w = _get_databricks_client()
            me = await asyncio.to_thread(w.current_user.me)
            status["databricks"] = {"user": me.user_name, "host": DATABRICKS_HOST}
        except Exception as e:
            status["databricks"] = {"error": str(e)}
    return JSONResponse(status)


async def parse_endpoint(request: Request):
    """Upload + parse in one step. Returns a document_id for MCP tool retrieval."""
    if not _has_databricks_creds:
        return JSONResponse({"error": "No Databricks credentials configured. Set DATABRICKS_HOST + DATABRICKS_CLIENT_ID/CLIENT_SECRET or DATABRICKS_TOKEN"}, status_code=500)

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

    if not _has_databricks_creds:
        return JSONResponse({"error": "No Databricks credentials configured. Set DATABRICKS_HOST + DATABRICKS_CLIENT_ID/CLIENT_SECRET or DATABRICKS_TOKEN"}, status_code=500)

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
        "client_id": "databricks-parser-public-client",
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
