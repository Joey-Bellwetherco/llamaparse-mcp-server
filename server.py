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
from starlette.responses import JSONResponse
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
        # Decode base64-encoded JSON credentials
        decoded = base64.b64decode(GOOGLE_DOCAI_CREDENTIALS_PATH)
        info = json.loads(decoded)
        creds = service_account.Credentials.from_service_account_info(info)
        return documentai.DocumentProcessorServiceClient(
            credentials=creds,
            client_options={"api_endpoint": endpoint},
        )
    else:
        # Fall back to default credentials (e.g. on GCP)
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
        # Extract header rows
        for header_row in table.header_rows:
            cells = []
            for cell in header_row.cells:
                text = _get_text_from_layout(cell.layout, document_text).strip()
                cells.append(text)
            rows.append("| " + " | ".join(cells) + " |")
            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        # Extract body rows
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

    # Force visual OCR on every page (no native text extraction)
    # This renders each page as an image and OCR's it, catching graphics,
    # unusual table layouts, charts, and visual elements that native parsing misses
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

    # Build output: full text + any extracted tables in markdown format
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
        # Images can't be split — send as-is
        chunks = [file_content]

    if len(chunks) == 1:
        return await asyncio.to_thread(_process_single_chunk, chunks[0], mime_type)

    # Process all chunks in parallel with a concurrency limit
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def process_with_limit(chunk: bytes) -> str:
        async with semaphore:
            return await asyncio.to_thread(_process_single_chunk, chunk, mime_type)

    results = await asyncio.gather(*[process_with_limit(c) for c in chunks])
    return "\n\n".join(results)


# In-memory store for uploaded files (keyed by upload ID)
_uploaded_files: dict[str, tuple[bytes, str, str]] = {}  # id -> (bytes, mime_type, filename)

mcp = FastMCP("Document AI MCP")


@mcp.tool()
async def parse_document_base64(
    document_base64: str,
    filename: str = "document.pdf",
    mime_type: str = "application/pdf",
) -> str:
    """Parse a document using Google Document AI with visual OCR.

    Send a base64-encoded document (PDF, image, etc.) and get back the parsed text
    with structured tables. Uses OCR to visually read every page, capturing graphics,
    charts, and unusual table formats.

    Args:
        document_base64: The document file content encoded as base64
        filename: Name of the file (for reference)
        mime_type: MIME type - application/pdf, image/png, image/jpeg, image/tiff, image/gif, image/bmp, image/webp
    """
    if not GOOGLE_DOCAI_CREDENTIALS_PATH:
        return "Error: No Google Cloud credentials configured. Set GOOGLE_DOCAI_CREDENTIALS_PATH."

    try:
        file_bytes = base64.b64decode(document_base64)
    except Exception:
        return "Error: Invalid base64 encoding. Please send a valid base64-encoded document."

    try:
        result = await _process_document(file_bytes, mime_type)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def parse_uploaded_document(
    upload_id: str,
) -> str:
    """Parse a previously uploaded document using Google Document AI Layout Parser.

    Use this after a file has been uploaded to the /upload endpoint.
    The upload_id is returned by the upload endpoint.

    Args:
        upload_id: The ID returned from the /upload endpoint
    """
    if not GOOGLE_DOCAI_CREDENTIALS_PATH:
        return "Error: No Google Cloud credentials configured. Set GOOGLE_DOCAI_CREDENTIALS_PATH."

    if upload_id not in _uploaded_files:
        return f"Error: No file found with upload_id '{upload_id}'. Upload a file first via POST /upload."

    file_bytes, mime_type, filename = _uploaded_files.pop(upload_id)

    try:
        result = await _process_document(file_bytes, mime_type)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def parse_document_from_url(
    url: str,
    mime_type: str = "application/pdf",
) -> str:
    """Parse a document from a URL using Google Document AI Layout Parser.

    Provide a URL to a document (PDF or image) and get back the parsed text content.

    Args:
        url: Direct URL to a document file
        mime_type: MIME type of the document (default: application/pdf)
    """
    if not GOOGLE_DOCAI_CREDENTIALS_PATH:
        return "Error: No Google Cloud credentials configured. Set GOOGLE_DOCAI_CREDENTIALS_PATH."

    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
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

    try:
        result = await _process_document(file_bytes, mime_type)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def check_status() -> str:
    """Check if the Document AI MCP server is configured and ready.

    Returns the server status and whether credentials are configured.
    """
    if GOOGLE_DOCAI_CREDENTIALS_PATH:
        return (
            f"Document AI MCP is running. Credentials configured. "
            f"Project: {GCP_PROJECT_ID}, Location: {GCP_LOCATION}, Processor: {GCP_PROCESSOR_ID}. "
            f"Ready to parse documents."
        )
    else:
        return "Document AI MCP is running but no credentials are configured. Set GOOGLE_DOCAI_CREDENTIALS_PATH."


# --- Starlette app with SSE transport + well-known endpoints for Claude.ai ---

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


async def oauth_protected_resource(request):
    """Tell clients no auth is required (MCP 2025-03-26 spec)."""
    return JSONResponse({
        "resource": f"https://{request.headers.get('host', 'localhost')}",
        "bearer_methods_supported": [],
        "resource_documentation": "https://github.com/Joey-Bellwetherco/llamaparse-mcp-server",
    })


async def oauth_authorization_server(request):
    """Return empty/minimal OAuth metadata since we don't require auth."""
    return JSONResponse({
        "issuer": f"https://{request.headers.get('host', 'localhost')}",
        "authorization_endpoint": f"https://{request.headers.get('host', 'localhost')}/authorize",
        "token_endpoint": f"https://{request.headers.get('host', 'localhost')}/token",
        "response_types_supported": ["code"],
    })


async def register(request):
    """Dynamic client registration endpoint — accept any registration."""
    body = await request.json()
    return JSONResponse({
        "client_id": "docai-public-client",
        "client_secret": "",
        "client_name": body.get("client_name", "Claude"),
        "redirect_uris": body.get("redirect_uris", []),
    })


async def upload_file(request: Request):
    """Upload a file for parsing. Returns an upload_id to use with the MCP tool."""
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
        # Raw binary upload
        file_bytes = await request.body()
        filename = request.headers.get("x-filename", "document.pdf")
        mime_type = content_type or "application/pdf"

    if not file_bytes:
        return JSONResponse({"error": "Empty file"}, status_code=400)

    upload_id = str(uuid.uuid4())
    _uploaded_files[upload_id] = (file_bytes, mime_type, filename)

    return JSONResponse({
        "upload_id": upload_id,
        "filename": filename,
        "size_bytes": len(file_bytes),
        "mime_type": mime_type,
        "message": f"File uploaded. Use the parse_uploaded_document tool with upload_id '{upload_id}' to parse it.",
    })


app = Starlette(
    routes=[
        Route("/health", health),
        Route("/upload", upload_file, methods=["POST"]),
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
