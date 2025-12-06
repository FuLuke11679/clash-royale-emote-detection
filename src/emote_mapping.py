"""
Emotion ↔ index mapping and emotion → emote mapping.

Assumes:
- Training data folders (and ImageFolder classes) are:
    angry, disgust, fear, happy, neutral, sad
- Emote images live in: data/emotes/<emotion>.<ext>
    e.g., data/emotes/angry.png, data/emotes/happy.jpg, etc.
"""

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IDX_TO_EMOTION = {i: name for i, name in enumerate(CLASS_NAMES)}
EMOTION_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# If your emote image files are literally named angry.png, happy.png, etc.,
# we can just map each emotion to the same string.
# (If you later want fancier Clash names, you can change values here.)
EMOTION_TO_EMOTE = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprise": "surprise"
}
