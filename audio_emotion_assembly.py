import sounddevice as sd
import numpy as np
import wavio
import requests
import time
import json

# ==============================
# CONFIGURATION
# ==============================
API_KEY = "f4eaded660134873a631dbf904f3a704"   # 🔑 Get from https://www.assemblyai.com/app
AUDIO_FILENAME = "recorded_audio.wav"
DURATION = 5          # seconds
SAMPLE_RATE = 44100
CHANNELS = 1

# ==============================
# STEP 1: RECORD AUDIO (with normalization + countdown)
# ==============================
print("🎤 Get ready! Recording starts in:")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

print("🎙️ Recording started! Speak clearly and naturally...")
audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float64')
sd.wait()

# Normalize volume
audio_data = audio_data / np.max(np.abs(audio_data))
wavio.write(AUDIO_FILENAME, audio_data, SAMPLE_RATE, sampwidth=2)
print(f"✅ Audio saved as '{AUDIO_FILENAME}'")

# ==============================
# STEP 2: UPLOAD AUDIO
# ==============================
print("⬆️ Uploading audio to AssemblyAI...")
headers = {"authorization": API_KEY}
upload_url = "https://api.assemblyai.com/v2/upload"

with open(AUDIO_FILENAME, "rb") as f:
    upload_response = requests.post(upload_url, headers=headers, data=f)
audio_url = upload_response.json()["upload_url"]
print("✅ Audio uploaded successfully!")

# ==============================
# STEP 3: REQUEST SENTIMENT / EMOTION ANALYSIS
# ==============================
print("🧠 Requesting sentiment & emotion analysis...")

transcript_request = {
    "audio_url": audio_url,
    "sentiment_analysis": True,
}
transcript_response = requests.post(
    "https://api.assemblyai.com/v2/transcript",
    json=transcript_request,
    headers=headers,
)
transcript_id = transcript_response.json()["id"]

# ==============================
# STEP 4: POLL UNTIL DONE
# ==============================
polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
print("⏳ Waiting for analysis to complete...")

while True:
    polling_response = requests.get(polling_endpoint, headers=headers)
    status = polling_response.json()["status"]

    if status == "completed":
        print("✅ Analysis complete!")
        break
    elif status == "error":
        print("❌ Error:", polling_response.json()["error"])
        exit()
    else:
        time.sleep(3)

# ==============================
# STEP 5: EXTRACT RESULTS
# ==============================
sentiments = polling_response.json().get("sentiment_analysis_results", [])

if not sentiments:
    print("⚠️ No sentiment data detected — possibly silent or unclear audio.")
    print("💡 Defaulting to 'Neutral' emotion.")
    final_scores = {
        "Happy": 0.0, "Neutral": 1.0, "Sad": 0.0,
        "Angry": 0.0, "Disgust": 0.0, "Fear": 0.0, "Surprise": 0.0
    }
    dominant_emotion = "Neutral"
else:
    # Aggregate sentiment scores
    score_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for s in sentiments:
        sentiment = s["sentiment"].lower()
        if sentiment in score_counts:
            score_counts[sentiment] += 1

    # Map sentiment → 7 emotions (approx.)
    emotion_mapping = {
        "positive": "Happy",
        "neutral": "Neutral",
        "negative": "Sad"
    }

    total = sum(score_counts.values())
    final_scores = {
        emotion_mapping[k]: round(v / total, 3)
        for k, v in score_counts.items() if v > 0
    }

    # Fill missing 7-class placeholders
    for e in ["Angry", "Disgust", "Fear", "Surprise"]:
        final_scores[e] = 0.0

    dominant_emotion = max(final_scores, key=final_scores.get)

# ==============================
# STEP 6: DISPLAY RESULTS
# ==============================
print("\n🎧 Audio Emotion Analysis Result:")
for emo, val in final_scores.items():
    print(f"  {emo}: {val*100:.1f}%")

print(f"\n🎯 Dominant Emotion: {dominant_emotion} 🧠")
