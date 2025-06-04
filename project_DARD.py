import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage
from fpdf import FPDF
import os
import sqlite3
from datetime import datetime
import pytz
import qrcode
import time
from streamlit_lottie import st_lottie
import json

# =============== Load models ===================
#model = tf.keras.models.load_model("trained_model.keras")
retinal_check_model = tf.keras.models.load_model("retinal_non_retinal_classifier.keras")
model = tf.keras.models.load_model("trained_model.keras", compile=False)

class_labels = ['0_No_DR', '1_Mild', '2_Moderate', '3_Severe', '4_Proliferate_DR']

IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

# Database
conn = sqlite3.connect("dard_reports.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, age INTEGER, gender TEXT,
    date TEXT, time TEXT, address TEXT,
    hospital_name TEXT, doctor_name TEXT,
    diagnosis TEXT, confidence REAL,
    image_path TEXT, pdf_path TEXT
)""")
conn.commit()

# =============== Load Lottie Animations ===================
def load_lottie_file(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return None

animation_welcome = load_lottie_file("theeye.json")
animation_processing = load_lottie_file("processing.json")
animation_success = load_lottie_file("done.json")

# =============== Functions ===================
def classify_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    return class_labels[predicted_class], confidence

def is_retinal_image(pil_img):
    img = pil_img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = retinal_check_model.predict(img_array)[0][0]
    return prediction > 0.5

def generate_pdf(user_info, diagnosis, confidence, image_file, pdf_link):
    report = FPDF()
    report.add_page()
    report.set_draw_color(0, 0, 0)
    report.rect(10, 10, 190, 277)

    report.set_font("Arial", 'B', 18)
    report.set_text_color(0, 102, 204)
    report.cell(0, 10, user_info["Hospital_Name"], ln=True, align='C')

    report.set_font("Arial", 'I', 14)
    report.set_text_color(90, 78, 89)
    report.cell(0, 8, f"Doctor: {user_info['Doctor_Name']}", ln=True, align='C')

    report.set_draw_color(200, 200, 200)
    report.line(20, report.get_y() + 2, 190, report.get_y() + 2)
    report.ln(6)

    report.set_font("Arial", 'B', 16)
    report.cell(0, 10, "Diabetes-Associated Retinal Detection Report", ln=True, align='C')
    report.ln(4)

    report.set_font("Arial", size=12)
    info_left = [("Name", user_info["Name"]), ("Age", str(user_info["Age"])), ("Gender", user_info["Gender"])]
    info_right = [("Date", user_info["Date"]), ("Time", user_info["Time"]), ("Address", user_info["Address"])]
    for i in range(3):
        y = report.get_y()
        report.set_xy(20, y)
        report.cell(40, 8, f"{info_left[i][0]}:", 0, 0)
        report.cell(60, 8, info_left[i][1], 0, 0)
        report.set_xy(110, y)
        report.cell(30, 8, f"{info_right[i][0]}:", 0, 0)
        report.multi_cell(60, 8, info_right[i][1], 0)
    report.ln(5)

    temp_img_path = "temp_image.jpg"
    image_file.save(temp_img_path)
    report.image(temp_img_path, x=70, y=report.get_y(), w=60)
    os.remove(temp_img_path)

    report.ln(70)

    report.set_fill_color(230, 230, 250)
    report.set_text_color(0, 0, 0)
    report.set_font("Arial", 'B', 14)
    report.cell(0, 10, f"Diagnosis: {diagnosis}", ln=True, fill=True, align='C')
    report.set_font("Arial", '', 12)
    report.cell(0, 8, f"Confidence: {confidence:.2f}%", ln=True, align='C')

    report.ln(10)

    qr = qrcode.make(pdf_link)
    qr_path = "temp_qr.png"
    qr.save(qr_path)
    report.image(qr_path, x=20, y=report.get_y(), w=40)
    os.remove(qr_path)

    report.set_xy(140, report.get_y() + 75)
    report.cell(50, 8, "Doctor's Signature", ln=True)
    report.line(140, report.get_y() + 2, 190, report.get_y() + 2)

    report.set_y(-26)
    report.set_font("Arial", 'I', 8)
    report.set_text_color(255, 0, 0)
    report.cell(0, 5, "Disclaimer: This report is for informational purposes only.", ln=True, align='C')

    output_dir = "result_DARD"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{user_info['Name'].replace(' ', '_')}_{user_info['Date']}.pdf"
    pdf_path = os.path.join(output_dir, filename)
    report.output(pdf_path)
    return pdf_path

# =============== Streamlit UI ===================
st.set_page_config(page_title="DARD: Diabetes-Associated Retinal Detection", layout="wide")
# Adaptive background color based on theme
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: var(--background-color);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------- Welcome page logic ---------------
if "started" not in st.session_state:
    st.session_state.started = False

if "diagnosis_completed" not in st.session_state:
    st.session_state.diagnosis_completed = False

if not st.session_state.started:
    st_lottie(animation_welcome, height=300)
    st.markdown('<h1 style="text-align:center; color:#4A90E2;"> Welcome to DARD System</h1>', unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; color: #555;'>Diabetes Associated Retinal Detection</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        if st.button("Enter DARD System"):
            st.session_state.started = True
            st.rerun()

else:
    # ORIGINAL APP STARTS HERE (your full existing code)
    with st.sidebar.expander("🌐 **About System**", expanded=False):    
        st.markdown("""
    DARD is an AI-powered diagnostic tool designed to assist healthcare professionals in the early detection and classification of Diabetic Retinopathy (DR). 
                     
    - Our platform enables doctors to upload retinal images and instantly receive a detailed, downloadable report outlining the severity of any detected retinal changes.
    - We aim to support early intervention and improve patient outcomes by providing fast, accurate, and easy-to-use screening assistance.
        """)
    with st.sidebar.expander("🔍 **About the Model**", expanded=False):    
        st.markdown("""
    Our model is built using advanced deep learning techniques, specifically Convolutional Neural Networks (CNNs), trained on thousands of annotated retinal images.
    
    - Trained on diverse datasets from real-world diabetic retinopathy cases.
    - Capable of classifying retinal images into five stages: No DR, Mild, Moderate, Severe, and Proliferative DR.
    - Achieved high performance metrics with an emphasis on sensitivity and specificity.
    - Designed for generalization across varied image qualities and patient demographics.
        """)

    with st.sidebar.expander("📋 **Diagnosis Summary Guide**", expanded=False):
        st.markdown("""
    ### 🩺 0 - No_DR
    - **Result:** No signs of diabetic retinopathy detected.
    - **Advice:** Routine monitoring recommended. Schedule next screening in 12 months unless symptoms develop.
    - **Clinical Observation:** No abnormalities such as microaneurysms, hemorrhages, or exudates were detected.
    
    ### 🩺 1 - Mild
    - **Result:** Early signs of diabetic retinopathy detected (Mild non-proliferative DR).
    - **Advice:** Recommend follow-up in 6-12 months. Control blood sugar levels and monitor eye health.
    - **Clinical Observation:** Presence of microaneurysms observed.
    
    ### 🩺 2 - Moderate
    - **Result:** Moderate non-proliferative diabetic retinopathy detected.
    - **Advice:** Closer monitoring advised. Follow-up within 3-6 months. May require treatment depending on progression.
    - **Clinical Observation:** Microaneurysms and hemorrhages detected.
    
    ### 🩺 3 - Severe
    - **Result:** Severe non-proliferative diabetic retinopathy detected.
    - **Advice:** High risk of progression. Immediate referral to an ophthalmologist is recommended.
    - **Clinical Observation:** Multiple hemorrhages and cotton wool spots observed.
    
    ### 🩺 4 - Proliferative_DR
    - **Result:** Proliferative diabetic retinopathy detected (advanced stage).
    - **Advice:** Urgent referral needed. High risk of vision loss. Immediate intervention required.
    - **Clinical Observation:** Neovascularization and vitreous hemorrhage signs detected.
        """)
    
    with st.sidebar.expander("⚠️ **Disclaimer**", expanded=False):    
        st.markdown("""
    - DARD provides <b>diagnostic support, not a final medical diagnosis.
    - Clinical judgment by a qualified doctor remains essential.
        """)

    # Main horizontal layout
    empty, col_left, col_right = st.columns([1, 2, 1])
    
    with col_left:
        with st.form("patient_form", clear_on_submit=False):
            st.markdown('''
                <div style="background-color: #fff2cc; padding: 20px; border-radius: 12px; margin-top: 20px; margin-bottom: 20px;">
                <h4 style="color: #7F6000;">🩺 Doctor Information</h4>
            </div>
            ''', unsafe_allow_html=True)
        
            #st.header("Hospital Details")
            hospital_name = st.text_input("Hospital Name")
            doctor_name = st.text_input("Doctor's Name")

            st.markdown('''
                <div style="background-color: #fff2cc; padding: 20px; border-radius: 12px;">
                  <h4 style="color: #0B5394;">🧑‍⚕️ Patient Information</h4>
                </div>
            ''', unsafe_allow_html=True)

            #st.subheader("Patient Information")
        
            name = st.text_input("Patient Name")
            age = st.number_input("Age", min_value=1, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            date = st.date_input("Date", now_ist.date()).strftime("%Y-%m-%d")
            time_input = st.time_input("Time", now_ist.time())
            time_str = time_input.strftime("%H:%M:%S")
            address = st.text_area("Address")
            image_file = st.file_uploader("Upload Retinal Image", type=["jpg", "jpeg", "png"])

            #submitted = st.button("Diagnose & Generate Report")
            submitted = st.form_submit_button("Submit")

    with col_right:
        animation_placeholder = st.empty()

    if submitted:
        if not all([hospital_name, doctor_name, name, age, gender, date, time_str, address, image_file]):
            st.error("Please fill all fields and upload an image.")
        else:
            # Update session state to indicate that the diagnosis process has been triggered
            st.session_state.diagnosis_completed = True

            with animation_placeholder:
                st_lottie(animation_processing, height=150)

            #time.sleep(2)

            img = PILImage.open(image_file).convert("RGB")
            if not is_retinal_image(img):
                animation_placeholder.empty()
                st.error("Uploaded image is not a valid retinal scan. Please upload a retinal image.")
            else:
                animation_placeholder.empty()
                st.success("Retinal image verified.")

                progress = st.progress(0)
                for i in range(100):
                    #time.sleep(0.01)
                    progress.progress(i + 1)

                diagnosis, confidence = classify_image(img)

                if diagnosis == '0_No_DR':
                    st.success(f"Diagnosis: **{diagnosis}**")
                elif diagnosis in ['1_Mild', '2_Moderate']:
                    st.warning(f"Diagnosis: **{diagnosis}**")
                else:
                    st.error(f"Diagnosis: **{diagnosis}**")
                st.info(f"Confidence: **{confidence:.2f}%**")

                st_lottie(animation_success, height=120)

                # Show the summary on the third page once diagnosis is completed
                if st.session_state.diagnosis_completed:
                    with st.expander("View Report Summary"):
                        st.write(f"**Hospital:** {hospital_name}")
                        st.write(f"**Doctor:** {doctor_name}")
                        st.write(f"**Patient Name:** {name}")
                        st.write(f"**Age:** {age}")
                        st.write(f"**Gender:** {gender}")
                        st.write(f"**Date:** {date}")
                        st.write(f"**Time:** {time_str}")
                        st.write(f"**Address:** {address}")
                        st.write(f"**Diagnosis:** {diagnosis}")
                        st.write(f"**Confidence:** {confidence:.2f}%")

                    # Proceed to generate PDF and database entry as usual
                    user_info = {
                        "Hospital_Name": hospital_name,
                        "Doctor_Name": doctor_name,
                        "Name": name,
                        "Age": age,
                        "Gender": gender,
                        "Date": date,
                        "Time": time_str,
                        "Address": address
                    }

                    pdf_filename = f"{name.replace(' ', '_')}_{date}.pdf"
                    pdf_link = os.path.join("result_DARD", pdf_filename)

                    with animation_placeholder:
                        st.lottie(animation_processing, height=100)

                    #time.sleep(1)

                    pdf_path = generate_pdf(user_info, diagnosis, confidence, img, pdf_link)

                    animation_placeholder.empty()

                    cursor.execute("""INSERT INTO reports (name, age, gender, date, time, address, hospital_name, doctor_name, diagnosis, confidence, image_path, pdf_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (name, age, gender, date, time_str, address, hospital_name, doctor_name, diagnosis, confidence, image_file.name, pdf_path))
                    conn.commit()

                    with open(pdf_path, "rb") as f:
                        st.download_button("Download Report", data=f, file_name=os.path.basename(pdf_path), mime="application/pdf")
