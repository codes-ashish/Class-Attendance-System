import sqlite3
import numpy as np

DB_NAME = "attendance.db"


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Creates tables. Also auto-migrates the old one-embedding-per-student
    schema (a single `embedding` column on `students`) into the new
    multi-embedding schema, so existing enrollments aren't lost."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
    needs_migration = False
    if cursor.fetchone():
        cursor.execute("PRAGMA table_info(students)")
        cols = [r[1] for r in cursor.fetchall()]
        needs_migration = "embedding" in cols  # old schema marker

    if needs_migration:
        cursor.execute("ALTER TABLE students RENAME TO students_legacy")
        conn.commit()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL,
            roll_no TEXT NOT NULL,
            name TEXT NOT NULL,
            UNIQUE(class_name, roll_no),
            FOREIGN KEY(class_name) REFERENCES classes(class_name) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
        )
    """)
    conn.commit()

    if needs_migration:
        cursor.execute("SELECT class_name, roll_no, name, embedding FROM students_legacy")
        for class_name, roll_no, name, blob in cursor.fetchall():
            cursor.execute(
                "INSERT OR IGNORE INTO students (class_name, roll_no, name) VALUES (?, ?, ?)",
                (class_name, roll_no, name),
            )
            cursor.execute(
                "SELECT id FROM students WHERE class_name = ? AND roll_no = ?",
                (class_name, roll_no),
            )
            row = cursor.fetchone()
            if row:
                cursor.execute(
                    "INSERT INTO embeddings (student_id, embedding) VALUES (?, ?)",
                    (row[0], blob),
                )
        cursor.execute("DROP TABLE students_legacy")
        conn.commit()

    conn.close()


# ----------------- CLASSES -----------------

def add_class(class_name):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO classes (class_name) VALUES (?)", (class_name.strip(),))
        conn.commit()
        return True, f"Class '{class_name}' created successfully!"
    except sqlite3.IntegrityError:
        return False, f"Class '{class_name}' already exists."
    finally:
        conn.close()


def delete_class(class_name):
    conn = get_connection()
    cursor = conn.cursor()
    # ON DELETE CASCADE takes care of students -> embeddings
    cursor.execute("DELETE FROM classes WHERE class_name = ?", (class_name,))
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
            "INSERT INTO students (class_name, roll_no, name) VALUES (?, ?, ?)",
            (class_name, roll_no.strip(), name.strip()),
        )
        student_id = cursor.lastrowid
        for emb in embeddings:
            blob = np.asarray(emb, dtype=np.float32).tobytes()
            cursor.execute(
                "INSERT INTO embeddings (student_id, embedding) VALUES (?, ?)",
                (student_id, blob),
            )
        conn.commit()
        return True, f"Student '{name}' added to {class_name} with {len(embeddings)} reference photo(s)!"
    except sqlite3.IntegrityError:
        return False, f"Roll No '{roll_no}' already exists in {class_name}."
    finally:
        conn.close()


def add_photo_to_student(class_name, roll_no, embedding):
    """Adds an extra reference embedding to an already-enrolled student."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM students WHERE class_name = ? AND roll_no = ?",
        (class_name, roll_no),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False, "Student not found."
    blob = np.asarray(embedding, dtype=np.float32).tobytes()
    cursor.execute(
        "INSERT INTO embeddings (student_id, embedding) VALUES (?, ?)",
        (row[0], blob),
    )
    conn.commit()
    conn.close()
    return True, "Reference photo added."


def delete_student(class_name, roll_no):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE class_name = ? AND roll_no = ?", (class_name, roll_no))
    conn.commit()
    conn.close()
    return True, f"Roll No {roll_no} removed from {class_name}."


def get_students_by_class(class_name):
    """Returns each student with a LIST of embeddings (one per reference photo)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, roll_no, name FROM students WHERE class_name = ? ORDER BY roll_no ASC",
        (class_name,),
    )
    rows = cursor.fetchall()
    students = []
    for sid, roll_no, name in rows:
        cursor.execute("SELECT embedding FROM embeddings WHERE student_id = ?", (sid,))
        embeddings = [np.frombuffer(b[0], dtype=np.float32) for b in cursor.fetchall()]
        students.append({"id": sid, "roll_no": roll_no, "name": name, "embeddings": embeddings})
    conn.close()
    return students
