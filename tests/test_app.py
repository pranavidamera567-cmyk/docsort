import io

import docx
import pytest

from app import create_app
from app.extract import ExtractionError, extract_text
from app.preprocess import preprocess

INVOICE = ("Invoice number 4471. Bill to Acme Corp. Amount due: subtotal 900, sales tax 90, total due 990. "
           "Payment terms net 30. Remit payment to our bank. Purchase order 88, unit price and quantity listed.")
RESUME = ("Professional summary. Work experience: led a team of engineers. Education: bachelor of science, university. "
          "Skills: proficient in Python. Certifications and internship. References available. Employment history.")


def make_pdf(text):
    """Smallest valid single-page PDF containing `text` (Helvetica)."""
    stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)


def make_docx(text):
    buf = io.BytesIO()
    d = docx.Document()
    d.add_paragraph(text)
    d.save(buf)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path):
    return create_app({"DATABASE": str(tmp_path / "t.db"), "TESTING": True}).test_client()


def upload(client, name, data):
    return client.post("/api/documents", data={"file": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_preprocess_drops_stopwords_digits_and_punctuation():
    assert preprocess("The Invoice #123 is OVERDUE, now!") == "invoice overdue"


def test_extraction_pdf_docx_txt_and_errors():
    assert "Invoice number" in extract_text("a.pdf", make_pdf("Invoice number 1"))
    assert "Work experience" in extract_text("a.docx", make_docx("Work experience here"))
    assert extract_text("a.txt", b"hello world") == "hello world"
    with pytest.raises(ExtractionError):
        extract_text("a.exe", b"x")
    with pytest.raises(ExtractionError):
        extract_text("a.txt", b"   ")
    with pytest.raises(ExtractionError):
        extract_text("a.pdf", b"not a pdf")


def test_upload_classifies_each_format(client):
    r = upload(client, "inv.pdf", make_pdf(INVOICE))
    assert r.status_code == 201
    doc = r.json["documents"][0]
    assert doc["category"] == "invoice" and doc["confidence"] > 0.4
    assert doc["confidence_level"] in ("high", "medium", "low")
    r = upload(client, "cv.docx", make_docx(RESUME))
    assert r.json["documents"][0]["category"] == "resume"
    assert upload(client, "bad.exe", b"x").status_code == 422


def test_search_ranking_filter_and_validation(client):
    upload(client, "inv.txt", INVOICE.encode())
    upload(client, "cv.txt", RESUME.encode())
    hits = client.get("/api/search?q=payment terms").json
    assert [h["filename"] for h in hits] == ["inv.txt"]
    assert "payment" in hits[0]["snippet"].lower() and hits[0]["score"] > 0
    assert client.get("/api/search?q=python").json[0]["filename"] == "cv.txt"
    assert client.get("/api/search?q=python&category=invoice").json == []
    assert client.get("/api/search?q=python&min_confidence=1.1").json == []
    assert client.get("/api/search").status_code == 400
    assert client.get("/api/search?q=zzzzqqq").json == []


def test_review_updates_category_and_status(client):
    doc = upload(client, "inv.txt", INVOICE.encode()).json["documents"][0]
    r = client.post(f"/api/documents/{doc['id']}/review", json={"category": doc["category"]})
    assert r.json["review_status"] == "approved"
    r = client.post(f"/api/documents/{doc['id']}/review", json={"category": "contract"})
    assert r.json["review_status"] == "corrected" and r.json["category"] == "contract"
    assert client.post(f"/api/documents/{doc['id']}/review", json={"category": "nope"}).status_code == 400
    assert client.get("/api/documents?category=contract").json[0]["id"] == doc["id"]


def test_custom_category_and_retrain_use_reviewed_documents(client):
    assert client.post("/api/categories", json={"name": "Meeting Minutes"}).status_code == 201
    minutes = "Meeting minutes. Attendees present. Agenda items discussed. Action items assigned. Next meeting scheduled."
    docs = [upload(client, f"m{i}.txt", minutes.encode()).json["documents"][0] for i in range(3)]
    for d in docs:
        client.post(f"/api/documents/{d['id']}/review", json={"category": "meeting_minutes"})
    assert client.post("/api/model/retrain").json["categories"] == 6
    new = upload(client, "m9.txt", minutes.encode()).json["documents"][0]
    assert new["category"] == "meeting_minutes"


def test_exports_and_admin_auth(tmp_path):
    app = create_app({"DATABASE": str(tmp_path / "a.db"), "TESTING": True, "ADMIN_PASSWORD": "secret"})
    c = app.test_client()
    upload(c, "inv.txt", INVOICE.encode())
    assert c.get("/admin").status_code == 401
    assert c.get("/api/export/summary.csv").status_code == 401
    assert c.get("/").status_code == 200  # search UI stays public
    import base64
    h = {"Authorization": "Basic " + base64.b64encode(b"admin:secret").decode()}
    csv_text = c.get("/api/export/summary.csv", headers=h).get_data(as_text=True)
    assert csv_text.splitlines()[0].startswith("category,documents,avg_confidence")
    assert "invoice" in csv_text
    assert c.get("/api/export/documents.csv", headers=h).status_code == 200
    assert c.get("/admin", headers=h).status_code == 200
