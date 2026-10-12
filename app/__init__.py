import csv
import functools
import io
import os
from pathlib import Path

import pandas as pd
from flask import Flask, Response, abort, jsonify, render_template, request

from . import db as database
from .classifier import Classifier, confidence_level
from .extract import ExtractionError, extract_text, file_type
from .search import search as run_search

ROOT = Path(__file__).resolve().parent.parent
EFFECTIVE = "COALESCE(final_category, predicted_category)"


def doc_json(r, include_text=False):
    d = {"id": r["id"], "filename": r["filename"], "file_type": r["file_type"], "title": r["title"],
         "category": r["final_category"] or r["predicted_category"],
         "predicted_category": r["predicted_category"], "confidence": round(r["confidence"], 3),
         "confidence_level": confidence_level(r["confidence"]), "review_status": r["review_status"],
         "uploaded_at": r["uploaded_at"], "char_count": r["char_count"]}
    if include_text:
        d["text"] = r["text"]
    return d


def create_app(config=None):
    app = Flask(__name__, template_folder="../templates")
    app.config.update(
        DATABASE=str(ROOT / "instance" / "docs.db"),
        TRAINING_CSV=str(ROOT / "data" / "training.csv"),
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
        ADMIN_PASSWORD=os.environ.get("ADMIN_PASSWORD"),
    )
    app.config.update(config or {})
    os.makedirs(os.path.dirname(app.config["DATABASE"]), exist_ok=True)
    database.init_db(app.config["DATABASE"])
    app.teardown_appcontext(database.close_db)
    clf = Classifier()

    def categories():
        return [r["name"] for r in database.get_db().execute("SELECT name FROM categories ORDER BY name")]

    def retrain():
        """Seed CSV plus admin-reviewed documents, restricted to configured categories."""
        db = database.get_db()
        valid = set(categories())
        seed = pd.read_csv(app.config["TRAINING_CSV"])
        reviewed = pd.read_sql_query(
            f"SELECT text, {EFFECTIVE} AS category FROM documents WHERE review_status != 'pending'", db)
        data = pd.concat([seed, reviewed], ignore_index=True)
        data = data[data["category"].isin(valid)]
        clf.train(data["text"].tolist(), data["category"].tolist())
        return {"training_examples": len(data), "categories": len(set(data["category"]))}

    with app.app_context():
        retrain()

    def admin_required(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            pw = app.config["ADMIN_PASSWORD"]
            if pw and not (request.authorization and request.authorization.password == pw):
                return Response("Admin login required", 401, {"WWW-Authenticate": 'Basic realm="docsort admin"'})
            return fn(*a, **kw)
        return wrapper

    # ---- pages -------------------------------------------------------------------------------
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/admin")
    @admin_required
    def admin():
        return render_template("admin.html")

    # ---- documents ---------------------------------------------------------------------------
    @app.post("/api/documents")
    def upload():
        files = request.files.getlist("file")
        if not files:
            return jsonify(error="No file uploaded (multipart field 'file')"), 400
        db, created, errors = database.get_db(), [], []
        for f in files:
            name = os.path.basename(f.filename or "")
            try:
                text = extract_text(name, f.read())
            except ExtractionError as e:
                errors.append({"filename": name, "error": str(e)})
                continue
            category, conf, _ = clf.predict(text)
            title = next((ln.strip() for ln in text.splitlines() if ln.strip()), name)[:120]
            cur = db.execute(
                "INSERT INTO documents (filename, file_type, title, text, char_count, predicted_category, confidence)"
                " VALUES (?,?,?,?,?,?,?)", (name, file_type(name), title, text, len(text), category, conf))
            created.append(doc_json(db.execute("SELECT * FROM documents WHERE id=?", (cur.lastrowid,)).fetchone()))
        db.commit()
        return jsonify(documents=created, errors=errors), (201 if created else 422)

    @app.get("/api/documents")
    def list_documents():
        sql, args = "SELECT * FROM documents WHERE 1=1", []
        if request.args.get("category"):
            sql += f" AND {EFFECTIVE} = ?"
            args.append(request.args["category"])
        if request.args.get("status"):
            sql += " AND review_status = ?"
            args.append(request.args["status"])
        rows = database.get_db().execute(sql + " ORDER BY uploaded_at DESC, id DESC", args).fetchall()
        return jsonify([doc_json(r) for r in rows])

    @app.get("/api/documents/<int:doc_id>")
    def get_document(doc_id):
        r = database.get_db().execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if r is None:
            abort(404)
        return jsonify(doc_json(r, include_text=True))

    @app.post("/api/documents/<int:doc_id>/review")
    @admin_required
    def review(doc_id):
        """Body {"category": "..."}: same as the prediction approves it, anything else corrects it."""
        db = database.get_db()
        r = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if r is None:
            abort(404)
        category = (request.get_json(silent=True) or {}).get("category") or r["predicted_category"]
        if category not in categories():
            return jsonify(error=f"Unknown category '{category}'"), 400
        status = "approved" if category == r["predicted_category"] else "corrected"
        db.execute("UPDATE documents SET review_status=?, final_category=? WHERE id=?", (status, category, doc_id))
        db.commit()
        return jsonify(doc_json(db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()))

    # ---- search ------------------------------------------------------------------------------
    @app.get("/api/search")
    def search():
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify(error="Missing query parameter 'q'"), 400
        min_conf = request.args.get("min_confidence", 0, type=float)
        rows = database.get_db().execute(f"SELECT *, {EFFECTIVE} AS category FROM documents").fetchall()
        hits = run_search(rows, q, request.args.get("category"))
        return jsonify([{**doc_json(r), "score": score, "snippet": snip}
                        for r, score, snip in hits if r["confidence"] >= min_conf])

    # ---- categories & model ------------------------------------------------------------------
    @app.get("/api/categories")
    def list_categories():
        rows = database.get_db().execute(
            "SELECT c.name, c.description, COUNT(d.id) AS documents FROM categories c "
            "LEFT JOIN documents d ON COALESCE(d.final_category, d.predicted_category) = c.name "
            "GROUP BY c.name ORDER BY c.name").fetchall()
        return jsonify([dict(r) for r in rows])

    @app.post("/api/categories")
    @admin_required
    def add_category():
        body = request.get_json(silent=True) or {}
        name = (body.get("name") or "").strip().lower().replace(" ", "_")
        if not name:
            return jsonify(error="'name' is required"), 400
        db = database.get_db()
        db.execute("INSERT OR IGNORE INTO categories VALUES (?, ?)", (name, body.get("description", "")))
        db.commit()
        return jsonify(name=name), 201

    @app.post("/api/model/retrain")
    @admin_required
    def retrain_endpoint():
        """Retrain on seed + reviewed data, then re-classify documents still awaiting review."""
        info = retrain()
        db = database.get_db()
        for r in db.execute("SELECT id, text FROM documents WHERE review_status='pending'").fetchall():
            cat, conf, _ = clf.predict(r["text"])
            db.execute("UPDATE documents SET predicted_category=?, confidence=? WHERE id=?", (cat, conf, r["id"]))
        db.commit()
        return jsonify(info)

    # ---- exports -----------------------------------------------------------------------------
    def csv_response(df, name):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={name}"})

    def documents_frame():
        return pd.read_sql_query(
            f"SELECT id, filename, file_type, uploaded_at, predicted_category, confidence, review_status, "
            f"{EFFECTIVE} AS final_category FROM documents ORDER BY id", database.get_db())

    @app.get("/api/export/documents.csv")
    @admin_required
    def export_documents():
        return csv_response(documents_frame(), "documents.csv")

    @app.get("/api/export/summary.csv")
    @admin_required
    def export_summary():
        df = documents_frame()
        if df.empty:
            return csv_response(pd.DataFrame(columns=["category", "documents"]), "classification_summary.csv")
        summary = df.groupby("final_category").agg(
            documents=("id", "size"),
            avg_confidence=("confidence", "mean"),
            low_confidence=("confidence", lambda s: int((s < 0.4).sum())),
            approved=("review_status", lambda s: int((s == "approved").sum())),
            corrected=("review_status", lambda s: int((s == "corrected").sum())),
            pending=("review_status", lambda s: int((s == "pending").sum())),
        ).round(3).reset_index().rename(columns={"final_category": "category"})
        return csv_response(summary, "classification_summary.csv")

    return app
