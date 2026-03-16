import os
import io
import json
import uuid
import base64
import asyncio
import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from mcp.server.sse import SseServerTransport
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account
from pypdf import PdfReader, PdfWriter

load_dotenv()

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

    # Force visual OCR on every page — renders pages as images and OCR's them
    process_options = documentai.ProcessOptions(
        ocr_config=documentai.OcrConfig(
            enable_native_pdf_parsing=False,
            language_code="en",
            premium_features=documentai.OcrConfig.PremiumFeatures(
                enable_selection_mark_detection=True,
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


# --- Parsed result cache ---
# POST /parse uploads + parses in one step, stores result by ID.
# MCP tool get_parsed_result retrieves by ID (tiny context footprint).
# Also supports page-range retrieval so Claude can fetch in batches.
_parsed_results: dict[str, dict] = {}  # id -> {"text": str, "pages": int, "filename": str}

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
        return "No parsed documents available. Upload a file via POST /parse first."

    lines = []
    for doc_id, entry in _parsed_results.items():
        lines.append(
            f"- {entry['filename']}: {entry['pages']} pages, "
            f"{len(entry['text'])} chars (id: {doc_id})"
        )
    return "Parsed documents:\n" + "\n".join(lines)


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


@mcp.tool()
async def check_status() -> str:
    """Check if the Document AI MCP server is configured and ready."""
    if GOOGLE_DOCAI_CREDENTIALS_PATH:
        return (
            f"Document AI MCP is running. Credentials configured. "
            f"Project: {GCP_PROJECT_ID}, Location: {GCP_LOCATION}, Processor: {GCP_PROCESSOR_ID}. "
            f"Ready to parse documents. "
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
        Route("/health", health),
        Route("/parse", parse_endpoint, methods=["POST"]),
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
