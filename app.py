"""
Person 5 module: Flask app that ties together URL analysis, ML prediction,
website/NLP analysis and DL prediction into one final risk assessment, with
a scan history stored in SQLite.
"""

import os
import sqlite3
import time
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify, g

from risk_engine import assess_url

DB_PATH = os.path.join(os.path.dirname(__file__), "scan_history.db")

app = Flask(__name__)


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            verdict TEXT NOT NULL,
            risk_percent REAL NOT NULL,
            ml_probability REAL,
            dl_probability REAL,
            website_probability REAL,
            scanned_at TEXT NOT NULL,
            duration_seconds REAL
        )
        """
    )
    try:
        conn.execute("ALTER TABLE scans ADD COLUMN duration_seconds REAL")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan", methods=["POST"])
def api_scan():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Please enter a URL"}), 400

    start = time.perf_counter()
    try:
        result = assess_url(url)
    except Exception as e:
        return jsonify({"error": f"Scan failed: {e}"}), 500
    duration = round(time.perf_counter() - start, 2)
    result["duration_seconds"] = duration

    db = get_db()
    db.execute(
        """INSERT INTO scans (url, verdict, risk_percent, ml_probability,
           dl_probability, website_probability, scanned_at, duration_seconds)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            result["url"], result["verdict"], result["risk_percent"],
            result["ml_probability"], result["dl_probability"],
            result["website_probability"],
            datetime.now(timezone.utc).isoformat(),
            duration,
        ),
    )
    db.commit()

    return jsonify(result)


@app.route("/api/history")
def api_history():
    db = get_db()
    rows = db.execute(
        "SELECT url, verdict, risk_percent, scanned_at, duration_seconds FROM scans "
        "ORDER BY id DESC LIMIT 25"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
