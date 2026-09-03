import os
from datetime import datetime

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis

import database as db
import face_utils as fu

# ----------------- CONFIGURATION & SECURITY -----------------
def get_admin_pin():
    """Prefers .streamlit/secrets.toml -> [admin_pin], then the
    ATTENDANCE_ADMIN_PIN env var, then falls back to the old hardcoded PIN."""
    try:
        if "admin_pin" in st.secrets:
            return str(st.secrets["admin_pin"])
    except Exception:
        pass
    return os.environ.get("ATTENDANCE_ADMIN_PIN", "1234")


PROFESSOR_PIN = get_admin_pin()
USING_DEFAULT_PIN = PROFESSOR_PIN == "1234"

MIN_RECOMMENDED_WIDTH = 1000  # px; below this, back-row faces are usually too small to help

st.set_page_config(page_title="Class Absentee Checker", layout="wide")
db.init_db()

if "is_authenticated" not in st.session_state:
    st.session_state["is_authenticated"] = False
if "enroll_photos" not in st.session_state:
    st.session_state["enroll_photos"] = []
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = None


@st.cache_resource
def load_face_model():
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(1280, 1280), det_thresh=0.35)
    return app


face_app = load_face_model()


def read_upload_to_bgr(uploaded_file):
    return cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def extract_main_embedding(img_bgr):
    """Enrollment photos are assumed to contain one primary subject; picks the largest face."""
    faces = face_app.get(img_bgr)
    if not faces:
        return None
    main = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return main.normed_embedding


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("🔒 Professor Access")
    if not st.session_state["is_authenticated"]:
        pin_input = st.text_input("Enter Passkey / PIN:", type="password")
        if st.button("Unlock Admin Controls", use_container_width=True):
            if pin_input == PROFESSOR_PIN:
                st.session_state["is_authenticated"] = True
                st.success("✅ Access Granted!")
                st.rerun()
            else:
                st.error("❌ Incorrect Passkey")
    else:
        st.success("🔓 Admin Mode Active")
        if st.button("Logout / Lock System", use_container_width=True):
            st.session_state["is_authenticated"] = False
            st.rerun()

    if USING_DEFAULT_PIN:
        st.caption(
            "⚠️ Using the default PIN. Set an `ATTENDANCE_ADMIN_PIN` environment "
            "variable, or `admin_pin` in `.streamlit/secrets.toml`, before real use."
        )

st.title("⚡ AI Class Absentee Checker")

classes = db.get_all_classes()

if st.session_state["is_authenticated"]:
    tab1, tab2, tab3 = st.tabs([
        "🔍 Check Absentees", "➕ Add Student (Admin)", "🏫 Manage Classes & Rosters (Admin)"
    ])
else:
    tab1, = st.tabs(["🔍 Check Absentees"])
    st.info("💡 Class/Student registration is locked. Unlock from sidebar.")

# ----------------- TAB 1: CHECK ABSENTEES -----------------
with tab1:
    if not classes:
        st.warning("No classes found. Unlock admin controls in the sidebar to create classes and add students.")
    else:
        active_class = st.selectbox("Choose Class:", classes, key="absent_checker_class")
        class_students = db.get_students_by_class(active_class)
        total_ref_photos = sum(len(s["embeddings"]) for s in class_students)
        st.caption(
            f"Total Enrolled in **{active_class}**: **{len(class_students)}** students "
            f"(**{total_ref_photos}** total reference photos)"
        )
        thin_students = [s["roll_no"] for s in class_students if len(s["embeddings"]) < 2]
        if thin_students:
            st.caption(
                f"💡 {len(thin_students)} student(s) have only one reference photo. Adding 2–3 more "
                "per student (different angle/lighting) in the Rosters tab meaningfully improves accuracy."
            )

        st.info(
            "📸 For best back-row detection, capture **one or more photos** — e.g. one from the "
            "front of the room and one from the back or a side aisle. Results are merged automatically, "
            "so a student only needs to be clearly visible in at least one photo."
        )

        class_source = st.radio(
            "Classroom Photo Source:",
            ["Upload Photo(s) from Device", "Take Photo with Camera"],
            horizontal=True,
        )

        classroom_frames = []
        if class_source == "Upload Photo(s) from Device":
            files_in = st.file_uploader(
                "Upload Classroom Photo(s) — higher resolution gives much better back-row detection",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key="classroom_files",
            )
            if files_in:
                classroom_frames = [read_upload_to_bgr(f) for f in files_in]
        else:
            cam_in = st.camera_input("Capture Classroom Photo", key="classroom_cam")
            if cam_in:
                classroom_frames = [read_upload_to_bgr(cam_in)]

        for i, frame in enumerate(classroom_frames):
            if frame.shape[1] < MIN_RECOMMENDED_WIDTH:
                st.warning(
                    f"Photo {i + 1} is only {frame.shape[1]}px wide — back-row faces may be too small "
                    f"to recognize no matter how the software processes it. {MIN_RECOMMENDED_WIDTH}px+ "
                    "width is recommended."
                )

        if classroom_frames and st.button("⚡ Scan for Absent Students", type="primary"):
            if not class_students:
                st.warning("No students enrolled in this class to match against.")
            else:
                with st.spinner("Enhancing image, scanning tiles, and refining distant faces…"):
                    combined_matches = {}
                    combined_review = {}
                    annotated_images = []
                    total_candidates = 0

                    for frame in classroom_frames:
                        detected = fu.detect_faces(face_app, frame)
                        total_candidates += len(detected)
                        matches, review_queue, unmatched = fu.match_faces(detected, class_students)

                        for roll_no, info in matches.items():
                            if roll_no not in combined_matches or info["similarity"] > combined_matches[roll_no]["similarity"]:
                                combined_matches[roll_no] = info

                        for r in review_queue:
                            if r["roll_no"] not in combined_matches:
                                if r["roll_no"] not in combined_review or r["similarity"] > combined_review[r["roll_no"]]["similarity"]:
                                    combined_review[r["roll_no"]] = r

                        annotated = fu.draw_annotations(frame, matches, review_queue, unmatched)
                        annotated_images.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

                    # A roll confirmed via one photo shouldn't linger in review from another.
                    combined_review = {rn: r for rn, r in combined_review.items() if rn not in combined_matches}

                    st.session_state["scan_results"] = {
                        "class_name": active_class,
                        "students": class_students,
                        "matches": combined_matches,
                        "review": combined_review,
                        "annotated_images": annotated_images,
                        "total_candidates": total_candidates,
                    }

        results = st.session_state["scan_results"]
        if results and results["class_name"] == active_class:
            matches = results["matches"]
            review = results["review"]
            absent_list = [s for s in results["students"] if s["roll_no"] not in matches]

            st.divider()

            if results["annotated_images"]:
                with st.expander("🖼️ Annotated Photo(s) — 🟢 present · 🟠 needs review · 🔴 unrecognized face"):
                    cols = st.columns(min(2, len(results["annotated_images"])))
                    for idx, img in enumerate(results["annotated_images"]):
                        cols[idx % len(cols)].image(img, use_container_width=True)

            if review:
                st.warning(f"🟠 {len(review)} borderline match(es) need your confirmation:")
                for roll_no, r in list(review.items()):
                    c1, c2, c3 = st.columns([3, 1, 1])
                    c1.write(f"**{r['name']}** (Roll {roll_no}) — confidence {r['similarity']:.2f}")
                    if c2.button("✅ Confirm", key=f"confirm_{roll_no}"):
                        matches[roll_no] = r
                        del review[roll_no]
                        st.rerun()
                    if c3.button("❌ Not present", key=f"reject_{roll_no}"):
                        del review[roll_no]
                        st.rerun()
                st.divider()

            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                st.error(f"### ❌ Absent Students ({len(absent_list)})")
                if not absent_list:
                    st.success("🎉 All enrolled students were detected present!")
                else:
                    absent_df = pd.DataFrame(
                        [{"Roll No": s["roll_no"], "Name": s["name"]} for s in absent_list]
                    )
                    st.table(absent_df)
                    st.download_button(
                        "⬇️ Download Absentee List (CSV)",
                        absent_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{active_class}_absentees_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                    )

            with col_res2:
                st.success("### 👥 Scan Summary")
                st.write(f"• **Face Candidates Found:** {results['total_candidates']}")
                st.write(f"• **Confirmed Present:** {len(matches)}")
                st.write(f"• **Pending Review:** {len(review)}")
                st.write(f"• **Absent:** {len(absent_list)}")

                if st.button("🔄 Reset / Clear Result"):
                    st.session_state["scan_results"] = None
                    st.rerun()

# ----------------- PROTECTED ADMIN TABS -----------------
if st.session_state["is_authenticated"]:
    # ----------------- TAB 2: ENROLL STUDENT -----------------
    with tab2:
        st.subheader("Enroll Student")
        if not classes:
            st.warning("⚠️ Please create at least one class under the 'Manage Classes' tab first.")
        else:
            target_class = st.selectbox("Assign to Class:", classes, key="enroll_target_class")

            col_e1, col_e2 = st.columns(2)
            with col_e1:
                enroll_roll = st.text_input("Student Roll Number:")
                enroll_name = st.text_input("Student Full Name:")
                st.caption(
                    "💡 Add 2–5 reference photos per student — varied angles, distances, and lighting "
                    "make matching far more reliable, especially from the back of the room."
                )
                photo_source = st.radio(
                    "Photo Source:", ["Upload from Device", "Use Camera"],
                    horizontal=True, key="enroll_photo_source",
                )

                if photo_source == "Upload from Device":
                    uploaded_photos = st.file_uploader(
                        "Upload Student Face Photo(s)",
                        type=['jpg', 'jpeg', 'png'],
                        accept_multiple_files=True,
                        key="enroll_uploads",
                    )
                    pending_images = [read_upload_to_bgr(f) for f in uploaded_photos] if uploaded_photos else []
                else:
                    cam_photo = st.camera_input("Capture Student Face", key="enroll_cam")
                    cc1, cc2 = st.columns(2)
                    if cam_photo is not None and cc1.button("➕ Add This Angle"):
                        st.session_state["enroll_photos"].append(read_upload_to_bgr(cam_photo))
                        st.rerun()
                    if cc2.button("🗑️ Clear Captured Photos"):
                        st.session_state["enroll_photos"] = []
                        st.rerun()
                    st.caption(f"Captured so far: {len(st.session_state['enroll_photos'])} photo(s)")
                    pending_images = st.session_state["enroll_photos"]

            with col_e2:
                st.write("### Action")
                if pending_images:
                    st.write(f"**{len(pending_images)}** photo(s) ready to process.")
                if st.button("💾 Save & Register Student", use_container_width=True):
                    if not enroll_roll or not enroll_name or not pending_images:
                        st.error("Please fill all details and provide at least one photo.")
                    else:
                        embeddings, skipped = [], 0
                        for img in pending_images:
                            emb = extract_main_embedding(img)
                            if emb is not None:
                                embeddings.append(emb)
                            else:
                                skipped += 1
                        if not embeddings:
                            st.error("No clear face detected in any provided photo. Please try again.")
                        else:
                            ok, msg = db.save_student(target_class, enroll_roll, enroll_name, embeddings)
                            if ok:
                                extra = f" ({skipped} photo(s) skipped — no face found.)" if skipped else ""
                                st.success(msg + extra)
                                st.session_state["enroll_photos"] = []
                                st.rerun()
                            else:
                                st.error(msg)

    # ----------------- TAB 3: MANAGE CLASSES & ROSTERS -----------------
    with tab3:
        col_c1, col_c2 = st.columns([1, 1])

        with col_c1:
            st.subheader("Create Class")
            with st.form("create_class_form", clear_on_submit=True):
                new_class = st.text_input("New Class / Subject Name:")
                create_btn = st.form_submit_button("➕ Create Class")
                if create_btn and new_class:
                    ok, msg = db.add_class(new_class)
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.rerun()

        with col_c2:
            st.subheader("Delete Class")
            if classes:
                class_to_del = st.selectbox("Select Class to Remove:", classes, key="del_class_select")
                if st.button("🗑️ Delete Selected Class", type="primary"):
                    ok, msg = db.delete_class(class_to_del)
                    st.warning(msg)
                    st.rerun()
            else:
                st.info("No classes to manage yet.")

        st.divider()
        st.subheader("📋 View & Manage Enrolled Students")
        if classes:
            view_class = st.selectbox("Select Class to View Roster:", classes, key="view_class_select")
            roster = db.get_students_by_class(view_class)

            if roster:
                df_display = pd.DataFrame([
                    {"Roll No": s["roll_no"], "Name": s["name"], "Reference Photos": len(s["embeddings"])}
                    for s in roster
                ])
                st.dataframe(df_display, use_container_width=True)

                thin = [s["roll_no"] for s in roster if len(s["embeddings"]) < 2]
                if thin:
                    st.caption(f"⚠️ Only 1 reference photo: {', '.join(thin)} — add more below for better accuracy.")

                with st.expander("➕ Add reference photo(s) to an existing student"):
                    boost_roll = st.selectbox("Student:", [s["roll_no"] for s in roster], key="boost_roll")
                    boost_files = st.file_uploader(
                        "Additional photo(s)", type=['jpg', 'jpeg', 'png'],
                        accept_multiple_files=True, key="boost_files",
                    )
                    if st.button("💾 Add Photo(s)"):
                        if not boost_files:
                            st.error("Please choose at least one photo.")
                        else:
                            added, skipped = 0, 0
                            for f in boost_files:
                                emb = extract_main_embedding(read_upload_to_bgr(f))
                                if emb is not None:
                                    db.add_photo_to_student(view_class, boost_roll, emb)
                                    added += 1
                                else:
                                    skipped += 1
                            extra = f" Skipped {skipped} (no face found)." if skipped else ""
                            st.success(f"Added {added} photo(s).{extra}")
                            st.rerun()

                del_student_roll = st.selectbox(
                    "Remove a student by Roll No:", [s["roll_no"] for s in roster], key="del_student_roll",
                )
                if st.button("❌ Remove Student"):
                    db.delete_student(view_class, del_student_roll)
                    st.success(f"Removed roll number {del_student_roll}")
                    st.rerun()
            else:
                st.info(f"No students enrolled in '{view_class}' yet.")

        st.divider()
        st.subheader("💾 Backup / Restore Roster Database")
        st.caption(
            "If you deploy this app to a free host (Streamlit Community Cloud, etc.), the local "
            "database file usually gets wiped whenever the app restarts or redeploys — that would "
            "mean re-enrolling every student. Download a backup after enrolling students, and "
            "restore it any time the roster comes back empty."
        )
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if os.path.exists(db.DB_NAME):
                with open(db.DB_NAME, "rb") as f:
                    st.download_button(
                        "⬇️ Download Backup",
                        f.read(),
                        file_name=f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.db",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )
        with bcol2:
            restore_file = st.file_uploader("Restore from backup (.db)", type=["db"], key="restore_upload")
            if restore_file is not None:
                st.warning("This will REPLACE all current classes, students, and reference photos.")
                if st.button("⚠️ Confirm Restore", use_container_width=True):
                    with open(db.DB_NAME, "wb") as f:
                        f.write(restore_file.getvalue())
                    st.success("Roster restored. Reloading…")
                    st.cache_resource.clear()
                    st.rerun()
