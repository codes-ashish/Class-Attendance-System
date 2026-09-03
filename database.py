import os
import base64
import sqlite3
import tempfile

import numpy as np
import psycopg2
import psycopg2.errors
import streamlit as st


def _get_database_url():
    try:
        if "DATABASE_URL" in st.secrets:
            return st.secrets["DATABASE_URL"]
    except Exception:
        pass
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "No DATABASE_URL found. Set it in .streamlit/secrets.toml locally, "
            "or as a Secret named DATABASE_URL on Streamlit Cloud."
        )
    return url


def get_connection():
    return psycopg2.connect(_get_database_url(), sslmode="require")


def init_db():
    """Creates tables if they don't exist yet. Safe to call on every app start."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id SERIAL PRIMARY KEY,
            class_name TEXT UNIQUE NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id SERIAL PRIMARY KEY,
            class_name TEXT NOT NULL REFERENCES classes(class_name) ON DELETE CASCADE,
            roll_no TEXT NOT NULL,
            name TEXT NOT NULL,
            UNIQUE(class_name, roll_no)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
            embedding BYTEA NOT NULL
        )
    """)

    conn.commit()
    cur.close()
    conn.close()


# ----------------- CLASSES -----------------

def add_class(class_name):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO classes (class_name) VALUES (%s)", (class_name.strip(),))
        conn.commit()
        return True, f"Class '{class_name}' created successfully!"
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return False, f"Class '{class_name}' already exists."
    finally:
        conn.close()


def delete_class(class_name):
    conn = get_connection()
    cursor = conn.cursor()
    # ON DELETE CASCADE takes care of students -> embeddings
    cursor.execute("DELETE FROM classes WHERE class_name = %s", (class_name,))
    conn.commit()
    conn.close()
    return True, f"Class '{class_name}' and its students were removed."


def get_all_classes():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT class_name FROM classes ORDER BY class_name ASC")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


# ----------------- STUDENTS / EMBEDDINGS -----------------

def save_student(class_name, roll_no, name, embeddings):
    """Creates a student with one or more reference embeddings. Multiple
    enrollment photos (different angle/lighting/distance) make matching much
    more robust than a single portrait."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO students (class_name, roll_no, name) VALUES (%s, %s, %s) RETURNING id",
            (class_name, roll_no.strip(), name.strip()),
        )
        student_id = cursor.fetchone()[0]
        for emb in embeddings:
            blob = np.asarray(emb, dtype=np.float32).tobytes()
            cursor.execute(
                "INSERT INTO embeddings (student_id, embedding) VALUES (%s, %s)",
                (student_id, psycopg2.Binary(blob)),
            )
        conn.commit()
        return True, f"Student '{name}' added to {class_name} with {len(embeddings)} reference photo(s)!"
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return False, f"Roll No '{roll_no}' already exists in {class_name}."
    finally:
        conn.close()


def add_photo_to_student(class_name, roll_no, embedding):
    """Adds an extra reference embedding to an already-enrolled student."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM students WHERE class_name = %s AND roll_no = %s",
        (class_name, roll_no),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False, "Student not found."
    blob = np.asarray(embedding, dtype=np.float32).tobytes()
    cursor.execute(
        "INSERT INTO embeddings (student_id, embedding) VALUES (%s, %s)",
        (row[0], psycopg2.Binary(blob)),
    )
    conn.commit()
    conn.close()
    return True, "Reference photo added."


def delete_student(class_name, roll_no):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE class_name = %s AND roll_no = %s", (class_name, roll_no))
    conn.commit()
    conn.close()
    return True, f"Roll No {roll_no} removed from {class_name}."


def get_students_by_class(class_name):
    """Returns each student with a LIST of embeddings (one per reference photo)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, roll_no, name FROM students WHERE class_name = %s ORDER BY roll_no ASC",
        (class_name,),
    )
    rows = cursor.fetchall()
    students = []
    for sid, roll_no, name in rows:
        cursor.execute("SELECT embedding FROM embeddings WHERE student_id = %s", (sid,))
        embeddings = [np.frombuffer(bytes(b[0]), dtype=np.float32) for b in cursor.fetchall()]
        students.append({"id": sid, "roll_no": roll_no, "name": name, "embeddings": embeddings})
    conn.close()
    return students


# ----------------- BACKUP / RECOVERY -----------------
# The roster now lives in Postgres, so it survives restarts on its own — these
# are just an extra safety net, and a one-time path to recover any roster you
# already had backed up as an old local .db file before this migration.

def export_all():
    """Returns a JSON-serializable snapshot of every class/student/embedding."""
    data = {"classes": []}
    for cname in get_all_classes():
        students = get_students_by_class(cname)
        data["classes"].append({
            "class_name": cname,
            "students": [
                {
                    "roll_no": s["roll_no"],
                    "name": s["name"],
                    "embeddings": [
                        base64.b64encode(np.asarray(e, dtype=np.float32).tobytes()).decode("ascii")
                        for e in s["embeddings"]
                    ],
                }
                for s in students
            ],
        })
    return data


def import_from_export(data):
    """Imports a snapshot produced by export_all(). Skips classes/students that
    already exist rather than erroring out, so it's safe to re-run."""
    classes_added = students_added = photos_added = 0
    for c in data.get("classes", []):
        ok, _ = add_class(c["class_name"])
        if ok:
            classes_added += 1
        for s in c.get("students", []):
            embs = [np.frombuffer(base64.b64decode(b), dtype=np.float32) for b in s["embeddings"]]
            if not embs:
                continue
            ok, _ = save_student(c["class_name"], s["roll_no"], s["name"], embs)
            if ok:
                students_added += 1
                photos_added += len(embs)
    return classes_added, students_added, photos_added


def import_from_sqlite_backup(file_bytes):
    """Recovers data from an OLD local .db backup (from before this Postgres
    migration) — handles both the single-embedding and multi-embedding
    SQLite schemas this project has used."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    classes_added = students_added = photos_added = 0
    try:
        src = sqlite3.connect(tmp_path)
        scur = src.cursor()

        scur.execute("SELECT class_name FROM classes")
        for (cname,) in scur.fetchall():
            ok, _ = add_class(cname)
            if ok:
                classes_added += 1

        scur.execute("PRAGMA table_info(students)")
        cols = [r[1] for r in scur.fetchall()]

        if "embedding" in cols:
            # oldest schema: one embedding column directly on students
            scur.execute("SELECT class_name, roll_no, name, embedding FROM students")
            for class_name, roll_no, name, blob in scur.fetchall():
                emb = np.frombuffer(blob, dtype=np.float32)
                ok, _ = save_student(class_name, roll_no, name, [emb])
                if ok:
                    students_added += 1
                    photos_added += 1
        else:
            # multi-embedding schema: separate embeddings table
            scur.execute("SELECT id, class_name, roll_no, name FROM students")
            for sid, class_name, roll_no, name in scur.fetchall():
                scur.execute("SELECT embedding FROM embeddings WHERE student_id = ?", (sid,))
                embs = [np.frombuffer(b[0], dtype=np.float32) for b in scur.fetchall()]
                if not embs:
                    continue
                ok, _ = save_student(class_name, roll_no, name, embs)
                if ok:
                    students_added += 1
                    photos_added += len(embs)

        src.close()
    finally:
        os.remove(tmp_path)

    return classes_added, students_added, photos_added