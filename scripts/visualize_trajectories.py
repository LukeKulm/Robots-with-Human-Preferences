import streamlit as st
import os
import json
from glob import glob
import random
import time

CLIP_DIR = os.path.abspath("data/clips")
PREFS_FILE = os.path.abspath("data/preferences.json")

def relative_path(path):
    return os.path.relpath(path, start=os.getcwd())

def load_clips():
    left_clips = sorted(glob(os.path.join(CLIP_DIR, "left_clip_*.mp4")))
    right_clips = sorted(glob(os.path.join(CLIP_DIR, "right_clip_*.mp4")))
    # Only return valid non-empty files
    pairs = [(l, r) for l, r in zip(left_clips, right_clips)
             if os.path.getsize(l) > 1000 and os.path.getsize(r) > 1000]
    return pairs

def load_preferences():
    if os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "r") as f:
            return json.load(f)
    return []

def save_preferences(prefs):
    with open(PREFS_FILE, "w") as f:
        json.dump(prefs, f, indent=2)

# --- Main UI ---
st.title("Trajectory Comparison: Human Feedback")

clips = load_clips()
prefs = load_preferences()

# Use relative paths for compatibility across environments
seen = {(relative_path(p['left']), relative_path(p['right'])) for p in prefs}
unseen = [(l, r) for l, r in clips if (relative_path(l), relative_path(r)) not in seen]

if unseen:
    current_pair = unseen[0]
    iter_number = os.path.basename(current_pair[0]).split("_")[-1].replace(".mp4", "")
    st.markdown(f"**Iteration:** {iter_number}")

if not unseen:
    st.success("✅ All clips have been labeled!")
    st.stop()

left_path, right_path = random.choice(unseen)

st.write("### Which trajectory is better?")
cols = st.columns(2)

with cols[0]:
    st.video(relative_path(left_path), format='video/mp4', start_time=0)
    st.caption("Left")

with cols[1]:
    st.video(relative_path(right_path), format='video/mp4', start_time=0)
    st.caption("Right")

choice = st.radio("Your preference", ["Left", "Right", "Tie", "Can't Tell"])

if st.button("Submit"):
    prefs.append({
        "left": relative_path(left_path),
        "right": relative_path(right_path),
        "preference": choice
    })
    save_preferences(prefs)
    # st.experimental_rerun()
    st.rerun()
