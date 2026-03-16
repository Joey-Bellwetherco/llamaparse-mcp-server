import os
import asyncio
import base64
import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY", "")
LLAMA_PARSE_API_URL = "https://api.cloud.llamaindex.ai/api/parsing"

mcp = FastMCP("LlamaParse MCP")


async def _upload_and_parse(
    file_content: bytes,
    filename: str,
    result_type: str = "markdown",
    language: str = "en",
) -> str:
    """Upload a file to LlamaParse and wait for the parsed result."""
    headers = {
        "Authorization": f"Bearer {LLAMA_CLOUD_API_KEY}",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        upload_resp = await client.post(
            f"{LLAMA_PARSE_API_URL}/upload",
            headers=headers,
            files={"file": (filename, file_content, "application/pdf")},
            data={
                "result_type": result_type,
                "language": language,
            },
        )
        upload_resp.raise_for_status()
        job = upload_resp.json()
        job_id = job["id"]

        max_attempts = 120
        for _ in range(max_attempts):
            status_resp = await client.get(
                f"{LLAMA_PARSE_API_URL}/job/{job_id}",
                headers=headers,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()

            if status_data["status"] == "SUCCESS":
                result_resp = await client.get(
                    f"{LLAMA_PARSE_API_URL}/job/{job_id}/result/{result_type}",
                    headers=headers,
                )
                result_resp.raise_for_status()
                result_data = result_resp.json()

                pages = result_data.get(result_type, result_data.get("pages", []))
                if isinstance(pages, list):
                    return "\n\n---\n\n".join(
                        p.get(result_type, p.get("text", str(p))) for p in pages
                    )
                return str(pages)

            elif status_data["status"] == "ERROR":
                raise Exception(
                    f"LlamaParse job failed: {status_data.get('error', 'Unknown error')}"
                )

            await asyncio.sleep(2)

        raise Exception("LlamaParse job timed out after 4 minutes")


@mcp.tool()
async def parse_pdf_base64(
    pdf_base64: str,
    filename: str = "document.pdf",
    output_format: str = "markdown",
) -> str:
    """Parse a PDF document using LlamaParse.

    Send a base64-encoded PDF and get back the parsed content as markdown or text.

    Args:
        pdf_base64: The PDF file content encoded as base64
        filename: Name of the file (for LlamaParse metadata)
        output_format: Output format - "markdown" or "text"
    """
    if not LLAMA_CLOUD_API_KEY:
        return "Error: LLAMA_CLOUD_API_KEY is not set. Please configure it in the environment."

    try:
        file_bytes = base64.b64decode(pdf_base64)
    except Exception:
        return "Error: Invalid base64 encoding. Please send a valid base64-encoded PDF."

    try:
        result = await _upload_and_parse(file_bytes, filename, output_format)
        return result
    except httpx.HTTPStatusError as e:
        return f"Error from LlamaParse API: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def parse_pdf_from_url(
    url: str,
    output_format: str = "markdown",
) -> str:
    """Parse a PDF from a URL using LlamaParse.

    Provide a URL to a PDF and get back the parsed content as markdown or text.

    Args:
        url: Direct URL to a PDF file
        output_format: Output format - "markdown" or "text"
    """
    if not LLAMA_CLOUD_API_KEY:
        return "Error: LLAMA_CLOUD_API_KEY is not set. Please configure it in the environment."

    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            file_bytes = resp.content
    except Exception as e:
        return f"Error downloading PDF from URL: {str(e)}"

    filename = url.split("/")[-1].split("?")[0] or "document.pdf"
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    try:
        result = await _upload_and_parse(file_bytes, filename, output_format)
        return result
    except httpx.HTTPStatusError as e:
        return f"Error from LlamaParse API: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def check_status() -> str:
    """Check if the LlamaParse MCP server is configured and ready.

    Returns the server status and whether the API key is set.
    """
    if LLAMA_CLOUD_API_KEY:
        return "LlamaParse MCP is running and API key is configured. Ready to parse documents."
    else:
        return "LlamaParse MCP is running but LLAMA_CLOUD_API_KEY is not set. Please configure it."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    # Use SSE transport served via starlette/uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    from starlette.responses import JSONResponse
    from mcp.server.sse import SseServerTransport

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

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    uvicorn.run(app, host="0.0.0.0", port=port)
