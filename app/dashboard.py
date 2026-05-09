import os
import sys
import numpy as np
import pandas as pd
import torch
import streamlit as st

# -------- PATH SETUP FIRST --------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

sys.path.insert(0, SRC_DIR)

# -------- IMPORT AFTER PATH --------
from lstm_model import LSTMModel
from fusion_model import FusionModel
from bert_module import DistilBERTEmbedder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- TEXT GENERATION --------
def generate_behavior_text(row):
    parts = []

    if row["ScreenTime"] > 0.7:
        parts.append("high screen time")
    elif row["ScreenTime"] > 0.4:
        parts.append("moderate screen time")
    else:
        parts.append("low screen time")

    if row["AppSwitching"] > 0.7:
        parts.append("frequent app switching")
    elif row["AppSwitching"] > 0.4:
        parts.append("moderate app switching")
    else:
        parts.append("low app switching")

    if row["LateNightUsage"] > 0.6:
        parts.append("high late night usage")
    else:
        parts.append("controlled night usage")

    if row["SleepHours"] < 0.4:
        parts.append("poor sleep pattern")
    else:
        parts.append("healthy sleep pattern")

    if row["SocialMedia"] > 0.6:
        parts.append("heavy social media engagement")

    if row["Gaming"] > 0.6:
        parts.append("high gaming activity")

    if row["MoodScore"] < 0.4:
        parts.append("possible emotional imbalance")
    else:
        parts.append("stable mood pattern")

    return "User shows " + ", ".join(parts) + "."


# -------- INPUT PREPARATION --------
def prepare_input_dataframe(
    age, gender, occupation, screen_time, social_media, gaming, streaming,
    work_education, unlocks, app_switching, notifications, late_night_usage,
    sleep_hours, mood_score, physical_activity, heart_rate
):
    df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Occupation": occupation,
        "ScreenTime": screen_time,
        "SocialMedia": social_media,
        "Gaming": gaming,
        "Streaming": streaming,
        "WorkEducation": work_education,
        "Unlocks": unlocks,
        "AppSwitching": app_switching,
        "Notifications": notifications,
        "LateNightUsage": late_night_usage,
        "SleepHours": sleep_hours,
        "MoodScore": mood_score,
        "PhysicalActivity": physical_activity,
        "HeartRate": heart_rate
    }])

    # Feature engineering
    df["TouchFrequency"] = df["Unlocks"] * 3
    df["SessionInterval"] = df["ScreenTime"] / (df["Unlocks"] + 1)
    df["NightUsageRatio"] = df["LateNightUsage"] / (df["ScreenTime"] + 1)
    df["EngagementScore"] = df["SocialMedia"] + df["Gaming"] + df["Streaming"]

    return df


# -------- LOAD MODELS --------
@st.cache_resource
def load_models(input_size):
    lstm_model = LSTMModel(input_size=input_size).to(DEVICE)
    lstm_model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "lstm_model.pth"), map_location=DEVICE)
    )
    lstm_model.eval()

    embedder = DistilBERTEmbedder()

    fusion_model = FusionModel(
        lstm_feature_dim=32,
        bert_dim=768,
        num_classes=3
    ).to(DEVICE)
    fusion_model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "fusion_model.pth"), map_location=DEVICE)
    )
    fusion_model.eval()

    return lstm_model, embedder, fusion_model


# -------- PREDICTION --------
def predict_stage(input_df, behavior_text):
    feature_order = [
        "Age", "Gender", "Occupation", "ScreenTime", "SocialMedia", "Gaming",
        "Streaming", "WorkEducation", "Unlocks", "AppSwitching", "Notifications",
        "LateNightUsage", "SleepHours", "MoodScore", "PhysicalActivity", "HeartRate",
        "TouchFrequency", "SessionInterval", "NightUsageRatio", "EngagementScore"
    ]

    input_df = input_df[feature_order]

    seq = np.repeat(input_df.values, 10, axis=0)
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    input_size = seq.shape[2]
    lstm_model, embedder, fusion_model = load_models(input_size)

    with torch.no_grad():
        lstm_features = lstm_model(seq, return_features=True)
        bert_embedding = embedder.encode_texts([behavior_text]).to(DEVICE)
        output = fusion_model(lstm_features, bert_embedding)

        pred = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    return pred, probs


# -------- UI --------
st.set_page_config(page_title="Digital Addiction Predictor", layout="wide")

st.title("Advanced Neuro-Behavioral Digital Addiction Predictor")
st.info("First run may take time due to model loading.")

st.subheader("Enter normalized values (0–1)")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 0.0, 1.0, 0.5)
    gender = st.selectbox("Gender", [0.0, 1.0])
    occupation = st.selectbox("Occupation", [0.0, 0.5, 1.0])
    screen_time = st.slider("Screen Time", 0.0, 1.0, 0.5)
    social_media = st.slider("Social Media", 0.0, 1.0, 0.5)
    gaming = st.slider("Gaming", 0.0, 1.0, 0.4)
    streaming = st.slider("Streaming", 0.0, 1.0, 0.4)
    work_education = st.slider("Work/Education", 0.0, 1.0, 0.5)

with col2:
    unlocks = st.slider("Unlocks", 0.0, 1.0, 0.5)
    app_switching = st.slider("App Switching", 0.0, 1.0, 0.5)
    notifications = st.slider("Notifications", 0.0, 1.0, 0.5)
    late_night_usage = st.slider("Late Night Usage", 0.0, 1.0, 0.4)
    sleep_hours = st.slider("Sleep Hours", 0.0, 1.0, 0.6)
    mood_score = st.slider("Mood Score", 0.0, 1.0, 0.5)
    physical_activity = st.slider("Physical Activity", 0.0, 1.0, 0.5)
    heart_rate = st.slider("Heart Rate", 0.0, 1.0, 0.5)

if st.button("Predict Addiction Stage"):
    with st.spinner("Running prediction..."):
        input_df = prepare_input_dataframe(
            age, gender, occupation, screen_time, social_media, gaming, streaming,
            work_education, unlocks, app_switching, notifications, late_night_usage,
            sleep_hours, mood_score, physical_activity, heart_rate
        )

        behavior_text = generate_behavior_text(input_df.iloc[0])

        st.subheader("Generated Behavior Description")
        st.write(behavior_text)

        pred, probs = predict_stage(input_df, behavior_text)

    # Correct mapping
    stage_map = {
        0: "High",
        1: "Low",
        2: "Moderate"
    }

    st.subheader("Prediction Result")
    st.success(stage_map[pred])

    st.subheader("Confidence")
    st.write({
        "High": float(probs[0]),
        "Low": float(probs[1]),
        "Moderate": float(probs[2])
    })

    # Behavioral Risk Check
    risk_score = sum([
        screen_time > 0.7,
        app_switching > 0.7,
        late_night_usage > 0.6,
        sleep_hours < 0.4,
        social_media > 0.6,
        gaming > 0.6
    ])

    st.subheader("Behavioral Risk Check")

    if risk_score >= 4:
        st.warning("High behavioral risk detected")
    elif risk_score >= 2:
        st.info("Moderate behavioral risk")
    else:
        st.success("Low behavioral risk")

    st.subheader("Interpretation")

    if pred == 0:
        st.error("High addiction risk detected")
    elif pred == 1:
        st.success("Low addiction risk")
    else:
        st.warning("Moderate addiction risk")