import streamlit as st
import os
import json
from glob import glob
import random

CLIP_DIR = "data/clips"
PREFS_FILE = "data/preferences.json"

def load_clips():
    left_clips = sorted(glob(os.path.join(CLIP_DIR, "left_clip_*.mp4")))
    right_clips = sorted(glob(os.path.join(CLIP_DIR, "right_clip_*.mp4")))
    return list(zip(left_clips, right_clips))

def load_preferences():
    if os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "r") as f:
            return json.load(f)
    return []

def save_preferences(prefs):
    with open(PREFS_FILE, "w") as f:
        json.dump(prefs, f, indent=2)

# Main UI
st.title("Trajectory Comparison: Human Feedback")

clips = load_clips()
prefs = load_preferences()

# Pick a random, unlabeled pair
seen = {(p['left'], p['right']) for p in prefs}
unseen = [(l, r) for l, r in clips if (l, r) not in seen]
if not unseen:
    st.success("All clips have been labeled!")
    st.stop()

left_path, right_path = random.choice(unseen)

st.write("### Which trajectory is better?")
cols = st.columns(2)

with cols[0]:
    st.video(left_path, format='video/mp4')
    st.caption("Left")

with cols[1]:
    st.video(right_path, format='video/mp4')
    st.caption("Right")

choice = st.radio("Your preference", ["Left", "Right", "Tie", "Can't Tell"])

if st.button("Submit"):
    prefs.append({
        "left": left_path,
        "right": right_path,
        "preference": choice
    })
    save_preferences(prefs)
    st.experimental_rerun()
