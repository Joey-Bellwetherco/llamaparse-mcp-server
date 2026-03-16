import os
import json
import base64
import asyncio
import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
from mcp.server.sse import SseServerTransport
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account

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


def _process_document(file_content: bytes, mime_type: str = "application/pdf") -> str:
    """Send a document to Document AI and return the parsed text."""
    client = _get_docai_client()
    resource_name = client.processor_path(GCP_PROJECT_ID, GCP_LOCATION, GCP_PROCESSOR_ID)
    raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)
    request = documentai.ProcessRequest(name=resource_name, raw_document=raw_document)
    result = client.process_document(request=request)
    return result.document.text


mcp = FastMCP("Document AI MCP")


@mcp.tool()
async def parse_document_base64(
    document_base64: str,
    filename: str = "document.pdf",
    mime_type: str = "application/pdf",
) -> str:
    """Parse a document using Google Document AI Layout Parser.

    Send a base64-encoded document (PDF, image, etc.) and get back the parsed text content.

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
        result = await asyncio.to_thread(_process_document, file_bytes, mime_type)
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
        result = await asyncio.to_thread(_process_document, file_bytes, mime_type)
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


app = Starlette(
    routes=[
        Route("/health", health),
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
