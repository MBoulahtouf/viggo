# tests/test_app.py
from fastapi.testclient import TestClient
from viggo.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Viggo API. Visit /docs for API documentation."}

# You would add more comprehensive tests here, e.g., for PDF upload and query.
# For PDF upload, you'd need a dummy PDF file to send.
# Example (conceptual, requires a dummy PDF):
# def test_upload_pdf():
#     with open("path/to/your/dummy.pdf", "rb") as f:
#         response = client.post("/api/v1/upload", files={"file": ("dummy.pdf", f, "application/pdf")})
#     assert response.status_code == 200
#     assert "num_chunks_indexed" in response.json()
