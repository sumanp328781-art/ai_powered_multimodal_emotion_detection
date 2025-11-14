"""
Agentic Response System for Multimodal Emotion Detector.

Usage:
  - Import `suggest_responses` and call with final emotion string
  - Or use `fuse_and_respond(video_scores, audio_scores)` to fuse two score dicts
  - The agent prints recommendations, can open a YouTube search for a suggested song,
    and can speak the response using pyttsx3 (optional).

Requirements:
  pip install pyttsx3
  (webbrowser is in stdlib)
"""

import webbrowser
import random
import pyttsx3
from typing import Dict, Optional

# --------------------------
# Helper: fuse two modality score dicts
# Both video_scores and audio_scores should be dicts mapping the 7 emotions to floats (0..1)
# weights: how much to trust video vs audio
# --------------------------
def fuse_scores(video_scores: Dict[str, float], audio_scores: Dict[str, float],
                video_weight: float = 0.6, audio_weight: float = 0.4) -> Dict[str, float]:
    # Ensure keys match; produce normalized fused scores
    keys = set(video_scores.keys()) | set(audio_scores.keys())
    fused = {}
    for k in keys:
        v = video_scores.get(k, 0.0) * video_weight + audio_scores.get(k, 0.0) * audio_weight
        fused[k] = v
    # normalize
    s = sum(fused.values()) or 1.0
    fused = {k: v / s for k, v in fused.items()}
    return fused

# --------------------------
# Response database (simple, extendable)
# Each emotion -> list of songs (YouTube search URLs), quotes, exercises
# --------------------------
RESPONSE_DB = {
    "Happy": {
        "songs": [
            "https://www.youtube.com/results?search_query=upbeat+happy+music+playlist",
            "https://www.youtube.com/results?search_query=feel+good+pop+hits"
        ],
        "quotes": [
            "Happiness is a direction, not a place. — Sydney J. Harris",
            "The only way to do great work is to love what you do. — Steve Jobs"
        ],
        "exercise": [
            "Do a 2-minute celebratory dance — put on a favorite song and move!",
            "3 sets of jumping jacks (20 reps) to keep the energy up."
        ]
    },
    "Sad": {
        "songs": [
            "https://www.youtube.com/results?search_query=uplifting+happy+songs+to+lift+your+mood",
            "https://www.youtube.com/results?search_query=calming+acoustic+music"
        ],
        "quotes": [
            "This too shall pass. — Persian adage",
            "The sun himself is weak when he first rises, and gathers strength and courage as the day gets on."
        ],
        "exercise": [
            "Try a 5-minute breathing exercise: inhale 4s, hold 4s, exhale 6s, repeat 5 times.",
            "Try progressive muscle relaxation: tense and release each muscle group for 5–10s."
        ]
    },
    "Angry": {
        "songs": [
            "https://www.youtube.com/results?search_query=calming+instrumental+music",
            "https://www.youtube.com/results?search_query=soothing+ambient+music"
        ],
        "quotes": [
            "For every minute you remain angry, you give up sixty seconds of peace of mind. — Ralph Waldo Emerson",
            "Speak when you are angry and you will make the best speech you will ever regret. — Ambrose Bierce"
        ],
        "exercise": [
            "Try box-breathing: inhale 4s, hold 4s, exhale 4s, hold 4s — repeat 6 times.",
            "Go for a brisk 5-minute walk to change physiology and cool down."
        ]
    },
    "Disgust": {
        "songs": [
            "https://www.youtube.com/results?search_query=calming+music+relax",
            "https://www.youtube.com/results?search_query=soft+piano+music"
        ],
        "quotes": [
            "The best way out is always through. — Robert Frost",
            "Be gentle with yourself — growth takes time."
        ],
        "exercise": [
            "Grounding exercise: name 5 things you can see, 4 you can touch, 3 you can hear.",
            "30 seconds of diaphragmatic breathing (slow belly breaths)."
        ]
    },
    "Fear": {
        "songs": [
            "https://www.youtube.com/results?search_query=calming+nature+sounds",
            "https://www.youtube.com/results?search_query=soothing+meditation+music"
        ],
        "quotes": [
            "Do the thing you fear and the death of fear is certain. — Emerson",
            "Courage doesn't always roar. Sometimes courage is the quiet voice at the end of the day saying, 'I will try again tomorrow.'"
        ],
        "exercise": [
            "5 minutes of paced breathing: inhale 4s, exhale 6s, repeat.",
            "Grounding: press feet into the floor, feel support, name 3 safe facts about now."
        ]
    },
    "Surprise": {
        "songs": [
            "https://www.youtube.com/results?search_query=celebration+songs",
            "https://www.youtube.com/results?search_query=upbeat+pop+hits"
        ],
        "quotes": [
            "Life is full of surprises — that's the joy of it.",
            "Embrace the unexpected — it often brings new opportunities."
        ],
        "exercise": [
            "Take 30 seconds to smile — smiling releases feel-good chemicals.",
            "A quick 1-minute mobility routine: neck rolls, shoulder rolls, hip circles."
        ]
    },
    "Neutral": {
        "songs": [
            "https://www.youtube.com/results?search_query=relaxing+background+music",
            "https://www.youtube.com/results?search_query=focus+instrumental+music"
        ],
        "quotes": [
            "Every day is a new beginning. Take a deep breath and start again.",
            "Small steps every day lead to big changes over time."
        ],
        "exercise": [
            "Try a 2-minute stretch routine: reach up, touch toes, gentle twists.",
            "Do 5 minutes of mindful breathing to check in with yourself."
        ]
    }
}

# --------------------------
# Agent: create response based on dominant emotion
# Options: open_song (bool): open youtube search; speak (bool): use TTS
# --------------------------
def suggest_responses(dominant_emotion: str, open_song: bool = False, speak: bool = True):
    E = dominant_emotion if dominant_emotion in RESPONSE_DB else "Neutral"
    data = RESPONSE_DB[E]

    # Pick one item from each category
    song = random.choice(data["songs"])
    quote = random.choice(data["quotes"])
    exercise = random.choice(data["exercise"])

    # Compose reply
    reply_lines = [
        f"Detected emotion: {E}",
        "",
        "Suggested music (YouTube search):",
        f"  {song}",
        "",
        "Quick exercise / breathing suggestion:",
        f"  {exercise}",
        "",
        "Motivational quote:",
        f"  \"{quote}\"",
    ]

    reply_text = "\n".join(reply_lines)

    print("\n" + reply_text + "\n")

    # Optionally speak (TTS)
    if speak:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(f"It seems you are feeling {E}. I recommended a song, a short exercise and a quote.")
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS error] {e}")

    # Optionally open the song (YouTube search url)
    if open_song:
        try:
            webbrowser.open(song)
            print(f"[Opened in browser] {song}")
        except Exception as e:
            print(f"[Browser error] {e}")

    # Return details (for logging / UI)
    return {"emotion": E, "song": song, "exercise": exercise, "quote": quote}

# --------------------------
# Convenience helper: fuse + respond
# video_scores and audio_scores are dicts like {'Happy':0.6,'Sad':0.1,...}
# --------------------------
def fuse_and_respond(video_scores: Dict[str, float], audio_scores: Dict[str, float],
                     video_weight: float = 0.6, audio_weight: float = 0.4,
                     open_song: bool = False, speak: bool = True):
    fused = fuse_scores(video_scores, audio_scores, video_weight, audio_weight)
    # determine dominant
    dominant = max(fused, key=fused.get)
    # call agent
    return suggest_responses(dominant, open_song=open_song, speak=speak)

# --------------------------
# Quick CLI demo when run directly
# --------------------------
if __name__ == "__main__":
    # Demo: call with a sample emotion (replace this with your fused result)
    demo_emotion = "Sad"   # change as needed
    suggest_responses(demo_emotion, open_song=False, speak=True)
