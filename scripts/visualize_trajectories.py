import streamlit as st
import os
import json
import glob
import random
import time
from datetime import datetime

CLIP_DIR = os.path.abspath("data/clips")
PREFS_FILE = os.path.abspath("data/preferences.json")

def relative_path(path):
    return os.path.relpath(path, start=os.getcwd())

def load_clips():
    left_clips = sorted(glob.glob(os.path.join(CLIP_DIR, "left_clip_*.mp4")))
    right_clips = sorted(glob.glob(os.path.join(CLIP_DIR, "right_clip_*.mp4")))
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

def get_latest_videos():
    """Get the latest pair of trajectory videos"""
    left_clips = sorted(glob.glob("videos/left_clip_*.mp4"))
    right_clips = sorted(glob.glob("videos/right_clip_*.mp4"))
    
    if not left_clips or not right_clips:
        return None, None
    
    return left_clips[-1], right_clips[-1]

def save_preference(preference):
    """Save the preference to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preference_data = {
        "timestamp": timestamp,
        "preference": preference
    }
    
    # Create preferences directory if it doesn't exist
    os.makedirs("preferences", exist_ok=True)
    
    # Save preference
    with open(f"preferences/preference_{timestamp}.json", "w") as f:
        json.dump(preference_data, f)

# --- Main UI ---
st.title("Trajectory Comparison: Human Feedback")

clips = load_clips()
prefs = load_preferences()

# Get the most recent iteration number from the clips
if clips:
    latest_iter = max(int(os.path.basename(l).split("_")[-1].replace(".mp4", "")) 
                     for l, _ in clips)
    st.markdown(f"**Current Iteration:** {latest_iter}")
    
    # Filter clips to only show the current iteration
    current_clips = [(l, r) for l, r in clips 
                    if int(os.path.basename(l).split("_")[-1].replace(".mp4", "")) == latest_iter]
    
    # Use relative paths for compatibility across environments
    seen = {(relative_path(p['left']), relative_path(p['right'])) for p in prefs}
    unseen = [(l, r) for l, r in current_clips 
             if (relative_path(l), relative_path(r)) not in seen]
    
    if not unseen:
        st.success("✅ All clips for this iteration have been labeled!")
        st.stop()
    
    left_path, right_path = unseen[0]  # Show the first unlabeled pair
    
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
        st.rerun()
else:
    st.warning("No clips found. Please run the training pipeline first.")

def main():
    st.title("Trajectory Preference Labeling")
    
    # Initialize session state for tracking last seen videos
    if 'last_left_video' not in st.session_state:
        st.session_state.last_left_video = None
    if 'last_right_video' not in st.session_state:
        st.session_state.last_right_video = None
    
    # Create columns for video display
    col1, col2 = st.columns(2)
    
    # Auto-refresh every 2 seconds
    while True:
        left_video, right_video = get_latest_videos()
        
        # Only update if we have new videos
        if (left_video and right_video and 
            (left_video != st.session_state.last_left_video or 
             right_video != st.session_state.last_right_video)):
            
            with col1:
                st.subheader("Trajectory 1")
                st.video(left_video)
                if st.button("Prefer Trajectory 1", key="left"):
                    save_preference("Left")
                    st.success("Preference saved!")
            
            with col2:
                st.subheader("Trajectory 2")
                st.video(right_video)
                if st.button("Prefer Trajectory 2", key="right"):
                    save_preference("Right")
                    st.success("Preference saved!")
            
            # Add a tie option
            st.button("Trajectories are Equal", key="tie")
            if st.session_state.tie:
                save_preference("Tie")
                st.success("Preference saved!")
            
            # Update session state
            st.session_state.last_left_video = left_video
            st.session_state.last_right_video = right_video
        
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()
