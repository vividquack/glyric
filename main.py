import whisper
import torch
import os
from ddgs import DDGS
import requests
from io import BytesIO
from moviepy import ImageClip, AudioFileClip, CompositeVideoClip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)

print("GLYRIC ONLY SUPPORTS ENGLISH AUDIO ONLY\n")
audio_path = input("Enter the path to the audio file: ")

if not os.path.exists(audio_path):
    raise FileNotFoundError(f"The file {audio_path} does not exist.")

try:
    # transcribe the audio file
    result = model.transcribe(audio_path, task="transcribe")
    results = result["text"]

    # map each single word to its time stamps
    segments = result["segments"]
    word_timestamps = []
    for segment in segments:
        words = segment["text"].strip().split()
        num_words = len(words)
        if num_words == 0:
            continue
        segment_duration = segment["end"] - segment["start"]
        word_duration = segment_duration / num_words

        for i, word in enumerate(words):
            start_time = segment["start"] + i * word_duration
            if i < num_words - 1:
                end_time = segment["start"] + (i + 1) * word_duration
            else:
                end_time = segment["end"]  # last word extends to segment end
            word_timestamps.append({
                "word": word,
                "start": start_time,
                "end": end_time
            })

    # load each image for each word from duckduckgo
    words_images = {}
    for item in word_timestamps:
        word = item["word"]
        if word in words_images:
            continue
        results = list(DDGS().images(word, max_results=1))
        image_url = None
        for result in results:
            image_url = result.get("image")
            break
        words_images[word] = image_url

    # make a video with the images and the timestamps
    audio_clip = AudioFileClip(audio_path)
    image_clips = []
    for item in word_timestamps:
        word = item["word"]
        start = item["start"]
        end = item["end"]
        duration = end - start
        image_url = words_images[word]
        if image_url is None:
            continue

        response = requests.get(image_url, timeout=10)
        if "image" not in response.headers.get("Content-Type", ""):
            print(f"Skipping non-image URL: {image_url}")
            continue

        try:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img_array = np.array(img)
        except Exception as e:
            print(f"Failed to open image for word '{word}': {e}")
            continue

        # create the clip with new API methods
        img_clip = (
            ImageClip(img_array)
            .with_duration(duration)
            .with_start(start)
            .resized(height=480)
        )
        image_clips.append(img_clip)

    video_clip = CompositeVideoClip(image_clips, size=(854, 480)).with_duration(audio_clip.duration)
    video_clip = video_clip.with_audio(audio_clip)
    output_path = os.path.splitext(audio_path)[0] + "_glyric.mp4"
    video_clip.write_videofile(output_path, fps=24)
    print(f"Glyric video created at {output_path}")

finally:
    pass
