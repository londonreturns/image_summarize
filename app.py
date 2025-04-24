from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from PIL import Image
import torch
import tempfile
import json
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Set device
device = torch.device("cpu")
print(f"\n🔧 Using device: {device}\n")

# Optional: torch performance tuning for CPU
torch.set_num_threads(os.cpu_count())
torch.backends.cudnn.benchmark = True

# Load model + processor once
print("📦 Loading model... (this may take a while)")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl").to(device)
model.eval()
print("✅ Model loaded!")

# Keyframe extraction
def get_keyframes(video_path, threshold=90):
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    last_frame = None
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if last_frame is not None:
            diff = cv2.absdiff(gray, last_frame)
            non_zero_count = np.count_nonzero(diff)
            if non_zero_count > threshold * gray.size / 100:
                keyframes.append((frame_id, frame))

        last_frame = gray
        frame_id += 1

    cap.release()
    return keyframes

# Safe captioning wrapper
def caption_image(img: Image.Image, prompt: str):
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100)
    result = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {"raw": result}

# Initialize ThreadPoolExecutor with a suitable number of workers (threads)
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

@app.route('/summarize', methods=['POST'])
def summarize_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files['video']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_path = tmp.name
        video_file.save(video_path)

    frames = get_keyframes(video_path)
    os.remove(video_path)

    # Initialize categories
    people, actions, objects, scenes, summaries = [], [], [], [], []

    # Prompts
    prompts = {
        "people": "List all people or animals visible in this image as a JSON array.",
        "actions": "List all actions happening in this image as a JSON array.",
        "objects": "List all visible objects in this image as a JSON array.",
        "scenes": "Describe the setting or scene in this image.",
        "summaries": "Summarize what is happening in this frame."
    }

    print(f"🖼 Extracted {len(frames)} keyframes.")

    # Run inference concurrently for each frame's prompts
    future_to_frame = {}
    for frame_id, frame in frames:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Submit inference tasks for each prompt (people, actions, etc.)
        future_people = executor.submit(caption_image, img, prompts["people"])
        future_actions = executor.submit(caption_image, img, prompts["actions"])
        future_objects = executor.submit(caption_image, img, prompts["objects"])
        future_scenes = executor.submit(caption_image, img, prompts["scenes"])
        future_summary = executor.submit(caption_image, img, prompts["summaries"])

        future_to_frame[frame_id] = {
            "people": future_people,
            "actions": future_actions,
            "objects": future_objects,
            "scenes": future_scenes,
            "summary": future_summary
        }

    # Collect results from futures
    for frame_id, futures in future_to_frame.items():
        people.append(futures["people"].result())
        actions.append(futures["actions"].result())
        objects.append(futures["objects"].result())
        scenes.append(futures["scenes"].result())
        summaries.append(futures["summary"].result())

    # Middle frame for overall summary
    mid_frame = frames[len(frames) // 2][1]
    overall_summary = caption_image(
        Image.fromarray(cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB)),
        "Provide a complete JSON summary of the video with keys: 'people', 'actions', 'objects', 'scene'."
    )

    return jsonify({
        "people": people,
        "actions": actions,
        "objects": objects,
        "scenes": scenes,
        "frame_summaries": summaries,
        "overall_summary": overall_summary
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
