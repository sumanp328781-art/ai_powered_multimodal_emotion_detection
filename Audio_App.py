import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import requests
import librosa

ASSEMBLY_KEY = "7752c8d2b0444baeabdcba2f704326e5"

st.title("7-Class Speech Emotion Detector (No HuggingFace)")

duration = st.slider("Record duration (seconds)", 1, 10)

# -----------------------------
# Emotion Classifier Function
# -----------------------------
def classify_emotion(sentence, rms, pitch):
    s = sentence.lower()

    emo_dict = {
        "happy": ["happy", "excited", "great", "good", "wonderful", "fantastic", "joy", "glad", "nice"],
        "sad": ["sad", "down", "unhappy", "depressed", "miserable", "lonely", "upset"],
        "angry": ["angry", "mad", "furious", "irritated", "annoyed", "ridiculous", "unacceptable", "frustrating"],
        "fearful": ["afraid", "scared", "terrified", "fear", "worried", "nervous", "anxious"],
        "disgust": ["disgust", "disgusting", "gross", "nasty", "revolting", "repulsive"],
        "surprise": ["surprised", "shocked", "unexpected", "whoa", "wow", "amazed"],
        "neutral": ["okay","alright","fine","maybe","note","schedule","plan","continue","confirm","will","go","to"],
    }

    scores = {e: 0.0 for e in emo_dict}

    # TEXT KEYWORDS
    for emo, words in emo_dict.items():
        for w in words:
            if w in s:
                scores[emo] += 2.0

    # Text punctuation cues
    if "!" in s:
        scores["angry"] += 0.5
        scores["surprise"] += 0.3

    # -------------------------
    # ACOUSTIC RULES
    # -------------------------

    # HAPPY
    if rms > 0.015 and 180 < pitch < 260:
        scores["happy"] += 0.8

    # SAD
    if rms < 0.008 and pitch < 150:
        scores["sad"] += 0.7

    # ANGRY
    if rms > 0.03 and pitch < 210:
        scores["angry"] += 1.0

    # SURPRISE
    if pitch > 230 and rms > 0.02:
        scores["surprise"] += 1.0

    # If everything zero → neutral
    if max(scores.values()) == 0:
        return "neutral"

    # If all are weak → neutral
    if max(scores.values()) < 1:
        return "neutral"

    return max(scores, key=scores.get)


# -----------------------------
# RECORD + PROCESS
# -----------------------------
if st.button("Record"):
    st.write("Recording…")
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()

    write("input.wav", fs, audio)
    st.write("Recording complete.")

    # Acoustic features
    rms = float(np.sqrt(np.mean(audio**2)))

    pitch, _ = librosa.piptrack(y=audio, sr=fs)
    pitch_mean = float(np.mean(pitch[pitch > 0])) if np.any(pitch > 0) else 0

    st.write(f"Intensity (RMS): {rms:.4f}")
    st.write(f"Pitch Mean (Hz): {pitch_mean:.2f}")

    # -------------------------
    # Upload to AssemblyAI
    # -------------------------
    st.write("Uploading to AssemblyAI…")
    upload = requests.post(
        "https://api.assemblyai.com/v2/upload",
        headers={"authorization": ASSEMBLY_KEY},
        data=open("input.wav", "rb")
    )
    audio_url = upload.json()["upload_url"]

    # -------------------------
    # Transcription
    # -------------------------
    st.write("Transcribing…")
    req = {
        "audio_url": audio_url,
        "sentiment_analysis": True
    }

    response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json=req,
        headers={"authorization": ASSEMBLY_KEY}
    )

    transcript_id = response.json()["id"]

    # Poll
    while True:
        poll = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
            headers={"authorization": ASSEMBLY_KEY}
        ).json()

        if poll["status"] == "completed":
            break
        if poll["status"] == "error":
            st.error("Transcription failed.")
            st.stop()

    text = poll["text"]
    st.write("Transcript:")
    st.success(text)

    # -------------------------
    # FINAL EMOTION PREDICTION
    # -------------------------
    emotion = classify_emotion(text, rms, pitch_mean)

    st.subheader("Predicted Emotion:")
    st.success(emotion.upper())
