import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
from collections import deque, Counter

# Load pre-trained model and face detector
model = load_model("fer2013_emotion_model.h5")
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

print("📹 Press 'r' to start recording a 10-second video.")
print("Press 'q' anytime to quit.")

cap = cv2.VideoCapture(0)
recorded_filename = "recorded_input.avi"

recording = False
frames = []
fps = 20
start_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Real-time Emotion Detection', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r') and not recording:
        print("🎬 Recording started...")
        start_time = time.time()
        recording = True

    if recording:
        frames.append(frame)
        if time.time() - start_time >= 10:
            print("✅ Recording complete! Saved as 'recorded_input.avi'")
            height, width, _ = frame.shape
            out = cv2.VideoWriter(recorded_filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
            for f in frames:
                out.write(f)
            out.release()
            break

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- EMOTION DETECTION ON RECORDED VIDEO ----------------
print("🧠 Analyzing emotions...")

cap = cv2.VideoCapture(recorded_filename)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_file = "recorded_output.avi"
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

emotion_window = deque(maxlen=8)  # keeps last 8 predictions for smoothing
all_emotions = []  # store for summary

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi, verbose=0)[0]
            label = emotion_labels[prediction.argmax()]
            confidence = np.max(prediction)

            # store and smooth emotion output
            emotion_window.append(label)
            all_emotions.append(label)
            smoothed_label = Counter(emotion_window).most_common(1)[0][0]

            label_text = f"{smoothed_label} ({confidence * 100:.1f}%)"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
print("✅ Emotion detection finished! Output saved as 'recorded_output.avi'")

# ---------------- PLAY THE OUTPUT VIDEO ----------------
print("🎥 Playing the analyzed video...")
cap = cv2.VideoCapture(output_file)
fps_video = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps_video) if fps_video > 0 else 50  # match original playback speed

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Analyzed Emotion Video", frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- EMOTION SUMMARY ----------------
if all_emotions:
    dominant = Counter(all_emotions).most_common(1)[0][0]
    print("\n🧠 Emotion Summary:")
    for emotion, count in Counter(all_emotions).most_common():
        print(f"  {emotion}: {count}")
    print(f"\n🎯 Dominant Emotion: {dominant}")
else:
    print("No faces detected.")
print("👋 Playback ended.")
