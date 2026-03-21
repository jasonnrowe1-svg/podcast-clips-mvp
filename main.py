from fastapi.responses import HTMLResponse
from fastapi import FastAPI
import subprocess
import json
import os
import re
import time
import whisper
from fastapi import UploadFile, File
import shutil
import threading
import uuid
from fastapi.responses import FileResponse
import zipfile
import cv2

app = FastAPI()

VIDEO_PATH = "sample_folder/20260308_093218.mp4"
AUDIO_PATH = "output/extracted_audio.wav"
TRANSCRIPT_PATH = "transcripts/transcript.txt"
SEGMENTS_PATH = "transcripts/segments.json"
CLIPS_PATH = "output/clips_fast"
CAPTIONS_PATH = "captions"
CAPTIONED_CLIPS_PATH = "output/captioned_clips"
FINAL_CLIPS_PATH = "output/final_clips"
THUMBNAILS_PATH = "output/thumbnails"

CURRENT_VIDEO = None
LAST_UPLOAD_INFO = {}
JOBS = {}
RENDER_JOBS = {}

model = whisper.load_model("base")


@app.get("/transcribe")
def transcribe():
    os.makedirs("transcripts", exist_ok=True)

    result = model.transcribe(AUDIO_PATH)

    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(result["text"])

    with open(SEGMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, indent=2)

    return {
        "message": "Transcription complete",
        "segment_count": len(result["segments"])
    }

@app.get("/extract-audio")
def extract_audio():

    if CURRENT_VIDEO is None:
        return {"error": "No video uploaded yet"}

    os.makedirs("output", exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", CURRENT_VIDEO,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "output/extracted_audio.wav"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    return {
        "message": "Audio extracted",
        "audio_file": "output/extracted_audio.wav",
        "ffmpeg_return_code": result.returncode
    }

@app.get("/find-clips")
def find_clips():
    with open(SEGMENTS_PATH, "r", encoding="utf-8") as f:
        segments = json.load(f)

    clips = []

    for s in segments:
        duration = s["end"] - s["start"]

        if duration >= 5 and duration <= 60:
            clips.append({
                "start": s["start"],
                "end": s["end"],
                "duration": round(duration, 2),
                "text": s["text"]
            })

    return {
        "clip_candidates": clips,
        "count": len(clips)
    }

@app.get("/cut-all-clips-fast")
def cut_all_clips_fast():
    os.makedirs(CLIPS_PATH, exist_ok=True)

    with open(SEGMENTS_PATH, "r", encoding="utf-8") as f:
        segments = json.load(f)

    clips = []
    for s in segments:
        duration = s["end"] - s["start"]
        if duration >= 10 and duration <= 60:
            clips.append({
                "start": s["start"],
                "end": s["end"],
                "duration": round(duration, 2),
                "text": s["text"]
            })

    if not clips:
        return {"error": "No clip candidates found"}

    created_clips = []

    for index, clip in enumerate(clips, start=1):
        output_file = os.path.join(CLIPS_PATH, f"clip_{index}.mp4")

        command = [
            "ffmpeg",
            "-y",
            "-ss", str(clip["start"]),
            "-to", str(clip["end"]),
            "-i", CURRENT_VIDEO ,
            "-c", "copy",
            output_file
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        created_clips.append({
            "clip_number": index,
            "output_file": output_file,
            "start": clip["start"],
            "end": clip["end"],
            "duration": clip["duration"],
            "text": clip["text"],
            "ffmpeg_return_code": result.returncode
        })

    return {
        "message": "All clips created fast",
        "count": len(created_clips),
        "clips": created_clips
    }

@app.get("/make-captions-file")
def make_captions_file():
    os.makedirs(CAPTIONS_PATH, exist_ok=True)

    with open(SEGMENTS_PATH, "r", encoding="utf-8") as f:
        segments = json.load(f)

    clips = []
    for s in segments:
        duration = s["end"] - s["start"]
        if duration >= 10 and duration <= 60:
            clips.append({
                "start": s["start"],
                "end": s["end"],
                "duration": round(duration, 2),
                "text": s["text"]
            })

    if not clips:
        return {"error": "No clip candidates found"}

    first_clip = clips[0]
    caption_file = os.path.join(CAPTIONS_PATH, "clip_1.srt")

    clip_segments = []
    for s in segments:
        if s["start"] >= first_clip["start"] and s["end"] <= first_clip["end"]:
            clip_segments.append(s)

    def format_srt_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    with open(caption_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(clip_segments, start=1):
            start_time = seg["start"] - first_clip["start"]
            end_time = seg["end"] - first_clip["start"]

            f.write(f"{i}\n")
            f.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
            f.write(f"{seg['text'].strip()}\n\n")

    return {
        "message": "Captions file created",
        "caption_file": caption_file,
        "segment_count": len(clip_segments)
    }

@app.get("/burn-captions")
def burn_captions():
    os.makedirs(CAPTIONED_CLIPS_PATH, exist_ok=True)

    input_clip = os.path.join(CLIPS_PATH, "clip_1.mp4")
    subtitle_file = os.path.join(CAPTIONS_PATH, "clip_1.srt")
    output_clip = os.path.join(CAPTIONED_CLIPS_PATH, "clip_1_captioned.mp4")

    subtitle_file_ffmpeg = subtitle_file.replace("\\", "/").replace(":", "\\:")

    command = [
        "ffmpeg",
        "-y",
        "-i", input_clip,
        "-vf", f"subtitles='{subtitle_file_ffmpeg}'",
        "-c:a", "copy",
        output_clip
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    return {
        "message": "Captioned clip created",
        "output_file": output_clip,
        "ffmpeg_return_code": result.returncode,
        "stderr": result.stderr[-500:]
    }

@app.get("/make-vertical")
def make_vertical():

    os.makedirs("output/vertical_clips", exist_ok=True)

    input_clip = "output/captioned_clips/clip_1_captioned.mp4"
    output_clip = "output/vertical_clips/clip_1_vertical.mp4"

    command = [
        "ffmpeg",
        "-y",
        "-i", input_clip,
        "-vf", "crop=1080:1920",
        "-c:a", "copy",
        output_clip
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    return {
        "message": "Vertical clip created",
        "output_file": output_clip,
        "ffmpeg_return_code": result.returncode
    }

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):

    global CURRENT_VIDEO, LAST_UPLOAD_INFO

    start_upload = time.time()

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join("uploads", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    CURRENT_VIDEO = file_path

    upload_time = round(time.time() - start_upload, 2)
    file_size_bytes = os.path.getsize(file_path)

    LAST_UPLOAD_INFO = {
        "filename": file.filename,
        "file_path": file_path,
        "upload_time_seconds": upload_time,
        "file_size_bytes": file_size_bytes
    }

    return {
        "message": "Video uploaded",
        "filename": file.filename,
        "file_path": file_path,
        "upload_time_seconds": upload_time,
        "file_size_bytes": file_size_bytes
    }


@app.get("/process-video")
def process_video():

    if CURRENT_VIDEO is None:
        return {"error": "No video uploaded yet"}

    os.makedirs("output", exist_ok=True)
    os.makedirs("transcripts", exist_ok=True)
    os.makedirs(CLIPS_PATH, exist_ok=True)
    os.makedirs(THUMBNAILS_PATH, exist_ok=True)

    # Step 1: extract audio
    extract_command = [
        "ffmpeg",
        "-y",
        "-i", CURRENT_VIDEO,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        AUDIO_PATH
    ]

    extract_result = subprocess.run(extract_command, capture_output=True, text=True)

    if extract_result.returncode != 0:
        return {
            "error": "Audio extraction failed",
            "ffmpeg_return_code": extract_result.returncode,
            "stderr": extract_result.stderr[-500:]
        }

    # Step 2: transcribe
    result = model.transcribe(AUDIO_PATH)

    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(result["text"])

    with open(SEGMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, indent=2)

    # Step 3: find clip candidates
    clips = []
    for s in result["segments"]:
        duration = s["end"] - s["start"]

        if duration >= 10 and duration <= 60:
            clips.append({
                "start": s["start"],
                "end": s["end"],
                "duration": round(duration, 2),
                "text": s["text"]
            })

    if not clips:
        return {
            "message": "Processing complete, but no clip candidates found",
            "segment_count": len(result["segments"]),
            "clip_count": 0
        }

    # Step 4: cut all clips fast
    created_clips = []

    for index, clip in enumerate(clips, start=1):
        output_file = os.path.join(CLIPS_PATH, f"clip_{index}.mp4")

        cut_command = [
            "ffmpeg",
            "-y",
            "-ss", str(clip["start"]),
            "-to", str(clip["end"]),
            "-i", CURRENT_VIDEO,
            "-c", "copy",
            output_file
        ]

        cut_result = subprocess.run(cut_command, capture_output=True, text=True)

        created_clips.append({
            "clip_number": index,
            "output_file": output_file,
            "start": clip["start"],
            "end": clip["end"],
            "duration": clip["duration"],
            "text": clip["text"],
            "ffmpeg_return_code": cut_result.returncode
        })

    return {
        "message": "Video processed successfully",
        "uploaded_video": CURRENT_VIDEO,
        "segment_count": len(result["segments"]),
        "clip_count": len(created_clips),
        "clips": created_clips
    }


def get_video_duration(video_path):
    command = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    data = json.loads(result.stdout)

    return float(data["format"]["duration"])


def get_clip_length_targets(video_duration):
    if video_duration < 60:
        return 5, 20
    elif video_duration < 300:
        return 8, 30
    else:
        return 15, 45
    
def normalize_text_for_scoring(text):
    cleaned = text.strip()
    lower_text = cleaned.lower()
    words = re.findall(r"\b[\w'-]+\b", lower_text)
    return cleaned, lower_text, words


def has_heavy_repetition(words):
    if len(words) < 12:
        return False

    repeated_bigrams = 0
    seen = {}

    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        seen[bigram] = seen.get(bigram, 0) + 1

    for count in seen.values():
        if count >= 3:
            repeated_bigrams += 1

    return repeated_bigrams >= 1


def get_hook_pattern_score(lower_text):
    hook_patterns = [
        "the problem is",
        "here's why",
        "this is why",
        "what people don't realize",
        "what people do not realize",
        "most people",
        "the truth is",
        "the reality is",
        "the reason is",
        "the reason why",
        "the mistake",
        "the biggest mistake",
        "nobody talks about",
        "you know what",
        "the thing is",
        "if you're",
        "if you are",
        "why do",
        "why does",
        "how do",
        "how does",
        "what happens when",
        "the difference is",
        "it turns out",
        "what matters is",
        "the key is",
        "the hard part is"
    ]

    score = 0
    for pattern in hook_patterns:
        if pattern in lower_text:
            score += 8

    return score


def get_takeaway_pattern_score(lower_text):
    takeaway_patterns = [
        "which means",
        "that means",
        "so if",
        "so the",
        "that's why",
        "this means",
        "the takeaway",
        "the lesson",
        "the point is",
        "in other words",
        "because",
        "therefore"
    ]

    score = 0
    for pattern in takeaway_patterns:
        if pattern in lower_text:
            score += 5

    return score


def get_specificity_score(cleaned, words):
    score = 0

    number_matches = re.findall(r"\b\d+\b", cleaned)
    score += min(len(number_matches) * 3, 9)

    if len(words) >= 40:
        score += 4

    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio > 0.72:
        score += 6
    elif unique_ratio > 0.6:
        score += 3

    return score


def get_quality_penalty(cleaned, lower_text, words):
    penalty = 0

    filler_words = {"um", "uh", "like", "you know", "sort of", "kind of", "basically"}
    filler_count = 0

    for filler in filler_words:
        filler_count += lower_text.count(filler)

    penalty += min(filler_count * 2, 12)

    weird_char_count = sum(1 for ch in cleaned if ch in "�")
    penalty += weird_char_count * 6

    if has_heavy_repetition(words):
        penalty += 15

    if len(words) < 18:
        penalty += 10

    if cleaned.endswith(("...", ",", ";", ":")):
        penalty += 8

    return penalty    


def score_clip_text(text, duration):
    score = 0
    cleaned, lower_text, words = normalize_text_for_scoring(text)
    word_count = len(words)

    if word_count == 0:
        return 0

    # 1. Strong duration preference for short-form clips
    if 18 <= duration <= 35:
        score += 22
    elif 15 <= duration <= 45:
        score += 16
    elif 12 <= duration <= 50:
        score += 8
    else:
        score -= 8

    # 2. Good word count range for spoken clips
    if 35 <= word_count <= 90:
        score += 18
    elif 25 <= word_count <= 110:
        score += 10
    else:
        score -= 6

    # 3. Spoken density
    if duration > 0:
        word_density = word_count / duration
        if 2.2 <= word_density <= 4.8:
            score += 12
        elif 1.6 <= word_density <= 5.5:
            score += 6
        else:
            score -= 5

    # 4. Hook / curiosity / opinion patterns
    score += get_hook_pattern_score(lower_text)

    # 5. Takeaway / conclusion patterns
    score += get_takeaway_pattern_score(lower_text)

    # 6. Specificity / detail
    score += get_specificity_score(cleaned, words)

    # 7. Direct audience language
    if "you" in words or "your" in words:
        score += 6

    # 8. Strong punctuation / spoken finish
    if cleaned.endswith(("?", "!", ".")):
        score += 6
    else:
        score -= 4

    if "?" in cleaned:
        score += 6

    # 9. First-person storytelling tends to clip well
    storytelling_terms = {"i", "we", "my", "our", "me"}
    storytelling_hits = sum(1 for word in words if word in storytelling_terms)
    score += min(storytelling_hits * 1.5, 8)

    # 10. Contrast creates tension
    contrast_terms = {"but", "however", "instead", "except", "although", "though", "yet"}
    contrast_hits = sum(1 for word in words if word in contrast_terms)
    score += min(contrast_hits * 3, 9)

    # 11. Penalize low-quality transcript segments
    score -= get_quality_penalty(cleaned, lower_text, words)

    # 12. Boundary quality still matters
    if has_good_clip_boundaries(cleaned):
        score += 10
    else:
        score -= 12

    return round(score, 2)

def get_audio_duration(audio_path):
    command = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        audio_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    data = json.loads(result.stdout)

    return float(data["format"]["duration"])

def has_good_clip_boundaries(text):
    cleaned = text.strip()

    if not cleaned:
        return False

    words = cleaned.split()
    if not words:
        return False

    first_word = words[0].lower().strip(".,!?")
    last_word = words[-1].lower().strip(".,!?")

    weak_starts = ["and", "but", "so", "because", "then", "or", "if", "well", "also", "anyway"]
    weak_endings = ["and", "but", "so", "because", "then", "or", "if", "to", "of", "with"]

    if first_word in weak_starts:
        return False

    if last_word in weak_endings:
        return False

    if not cleaned.endswith((".", "!", "?")):
        return False

    if len(words) < 12:
        return False

    if "�" in cleaned:
        return False

    return True

def detect_main_face_center_x(video_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return None

    sample_points = [0.15, 0.35, 0.5, 0.65, 0.85]
    centers = []
    frame_width = None
    frame_height = None

    for point in sample_points:
        target_frame = int(frame_count * point)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        success, frame = cap.read()

        if not success or frame is None:
            continue

        frame_height, frame_width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            continue

        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        center_x = x + (w / 2)
        centers.append(center_x)

    cap.release()

    if not centers or frame_width is None or frame_height is None:
        return None

    avg_center_x = sum(centers) / len(centers)
    return avg_center_x, frame_width, frame_height

def build_face_centered_crop_filter(input_clip_path, output_width=720, output_height=1280):
    face_result = detect_main_face_center_x(input_clip_path)

    if face_result is None:
        return (
            f"scale={output_width}:{output_height}:force_original_aspect_ratio=increase,"
            f"crop={output_width}:{output_height}"
        )

    center_x, frame_width, frame_height = face_result

    # For vertical 9:16 output, crop width should be based on frame height
    crop_width = frame_height * (output_width / output_height)
    crop_width = min(crop_width, frame_width)

    left = max(0, min(center_x - (crop_width / 2), frame_width - crop_width))

    return (
        f"crop={int(crop_width)}:{int(frame_height)}:{int(left)}:0,"
        f"scale={output_width}:{output_height}"
    )


def run_pipeline_job(job_id, video_path):
    try:
        start_total = time.time()
        timings = {}        
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["stage"] = "Extracting audio..."
        JOBS[job_id]["progress"] = 15

        os.makedirs("output", exist_ok=True)
        os.makedirs("transcripts", exist_ok=True)
        os.makedirs(CLIPS_PATH, exist_ok=True)
        os.makedirs(THUMBNAILS_PATH, exist_ok=True)

        # extract audio
        start_audio = time.time()        
        extract_command = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            AUDIO_PATH
        ]

        extract_result = subprocess.run(extract_command, capture_output=True, text=True)
        timings["audio_extraction"] = round(time.time() - start_audio, 2)        

        if extract_result.returncode != 0:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["stage"] = "Processing failed"
            JOBS[job_id]["progress"] = 0
            JOBS[job_id]["error"] = "Audio extraction failed"
            return
        
        audio_duration = get_audio_duration(AUDIO_PATH)
        JOBS[job_id]["audio_duration"] = audio_duration

        if audio_duration < 1:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["stage"] = "Processing failed"
            JOBS[job_id]["progress"] = 0
            JOBS[job_id]["error"] = "Extracted audio is too short or empty"
            return

        JOBS[job_id]["stage"] = "Transcribing audio..."
        JOBS[job_id]["progress"] = 35
        
        # transcribe
        start_transcribe = time.time()
        result = model.transcribe(AUDIO_PATH)
        timings["transcription"] = round(time.time() - start_transcribe, 2)        

        with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
            f.write(result["text"])

        with open(SEGMENTS_PATH, "w", encoding="utf-8") as f:
            json.dump(result["segments"], f, indent=2)

        video_duration = get_video_duration(video_path)
        min_clip_duration, max_clip_duration = get_clip_length_targets(video_duration)

        JOBS[job_id]["video_duration"] = video_duration
        JOBS[job_id]["min_clip_duration"] = min_clip_duration
        JOBS[job_id]["max_clip_duration"] = max_clip_duration


        JOBS[job_id]["stage"] = "Building clip candidates..."
        JOBS[job_id]["progress"] = 55
        
        start_clip_gen = time.time()
        # find clips by combining neighboring segments
        clips = []
        segments = result["segments"]

        for i in range(len(segments)):
            start_time = segments[i]["start"]
            end_time = segments[i]["end"]
            combined_text = segments[i]["text"].strip()

            for j in range(i + 1, len(segments)):
                new_end = segments[j]["end"]
                new_duration = new_end - start_time

                if new_duration > max_clip_duration:
                    break

                end_time = segments[j]["end"]
                combined_text += " " + segments[j]["text"].strip()

                duration = end_time - start_time

                if duration >= (min_clip_duration * 0.7):
                    clip_text = combined_text.strip()

                    # don't accept weak boundaries too early
                    if not has_good_clip_boundaries(clip_text):
                        continue

                    # if we have room, keep extending to try to complete the thought
                    extended_end_time = end_time
                    extended_text = clip_text

                    for k in range(j + 1, len(segments)):
                        possible_end = segments[k]["end"]
                        possible_duration = possible_end - start_time

                        if possible_duration > max_clip_duration:
                            break

                        extended_end_time = segments[k]["end"]
                        extended_text += " " + segments[k]["text"].strip()

                        if has_good_clip_boundaries(extended_text.strip()):
                            clip_text = extended_text.strip()
                            end_time = extended_end_time

                    duration = end_time - start_time
                    clip_score = score_clip_text(clip_text, duration)

                    if clip_score < 55:
                        continue

                    clips.append({
                        "start": start_time,
                        "end": end_time,
                        "duration": round(duration, 2),
                        "text": clip_text,
                        "score": clip_score
                    })
                    break                
                
        timings["clip_generation"] = round(time.time() - start_clip_gen, 2) 
        
                       
        JOBS[job_id]["stage"] = "Scoring and filtering clips..."
        JOBS[job_id]["progress"] = 70
        clips.sort(key=lambda x: x["score"], reverse=True)
        JOBS[job_id]["raw_clip_candidates"] = len(clips)

        CANDIDATE_POOL_SIZE = 30
        FINAL_MAX_CLIPS = 8

        clips = clips[:CANDIDATE_POOL_SIZE]

        # keep best non-overlapping clips
        filtered_clips = []
        remaining_clips = clips.copy()

        while remaining_clips and len(filtered_clips) < FINAL_MAX_CLIPS:
            best_clip = remaining_clips.pop(0)
            filtered_clips.append(best_clip)

            non_overlapping = []
            for clip in remaining_clips:
                overlaps = not (
                    clip["end"] <= best_clip["start"] or
                    clip["start"] >= best_clip["end"]
                )

                if not overlaps:
                    non_overlapping.append(clip)

            remaining_clips = non_overlapping

        filtered_clips.sort(key=lambda x: x["start"])
        clips = filtered_clips
        JOBS[job_id]["filtered_clip_candidates"] = len(clips)
        JOBS[job_id]["stage"] = "Cutting clips and generating thumbnails..."
        JOBS[job_id]["progress"] = 85

        start_cutting = time.time()
        created_clips = []

        for index, clip in enumerate(clips, start=1):
            output_file = os.path.join(CLIPS_PATH, f"{job_id}_clip_{index}.mp4")
            thumbnail_file = os.path.join(THUMBNAILS_PATH, f"{job_id}_clip_{index}.jpg")
            
            JOBS[job_id]["stage"] = f"Cutting clip {index} of {len(clips)}..."
            JOBS[job_id]["progress"] = min(95, 85 + int((index / max(1, len(clips))) * 8))

            cut_command = [
                "ffmpeg",
                "-y",
                "-ss", str(clip["start"]),
                "-to", str(clip["end"]),
                "-i", video_path,
                "-c", "copy",
                output_file
            ]

            cut_result = subprocess.run(cut_command, capture_output=True, text=True)
            
            JOBS[job_id]["stage"] = f"Generating thumbnail {index} of {len(clips)}..."
            JOBS[job_id]["progress"] = min(98, 86 + int((index / max(1, len(clips))) * 10))

            thumbnail_command = [
                "ffmpeg",
                "-y",
                "-ss", "00:00:00.5",
                "-i", output_file,
                "-vframes", "1",
                thumbnail_file
            ]

            thumbnail_result = subprocess.run(thumbnail_command, capture_output=True, text=True)
            thumbnail_success = thumbnail_result.returncode == 0 and os.path.exists(thumbnail_file)

            created_clips.append({
                "clip_number": index,
                "output_file": output_file,
                "thumbnail_file": thumbnail_file if thumbnail_success else None,
                "start": clip["start"],
                "end": clip["end"],
                "duration": clip["duration"],
                "text": clip["text"],
                "score": clip["score"],
                "ffmpeg_return_code": cut_result.returncode
            })

        timings["cutting_and_thumbnails"] = round(time.time() - start_cutting, 2)
        timings["total"] = round(time.time() - start_total, 2)

        JOBS[job_id]["status"] = "complete"
        JOBS[job_id]["stage"] = "Ready"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["segment_count"] = len(result["segments"])
        JOBS[job_id]["clip_count"] = len(created_clips)
        JOBS[job_id]["clips"] = created_clips
        JOBS[job_id]["timings"] = timings        
        

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["stage"] = "Processing failed"
        JOBS[job_id]["progress"] = 0
        JOBS[job_id]["error"] = str(e)

@app.post("/start-job")
def start_job():

    if CURRENT_VIDEO is None:
        return {"error": "No video uploaded yet"}

    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "status": "processing",
        "stage": "Starting job...",
        "progress": 5,
        "clips": [],
        "upload_info": LAST_UPLOAD_INFO.copy()
    }
    

    thread = threading.Thread(target=run_pipeline_job, args=(job_id, CURRENT_VIDEO))
    thread.start()

    return {
        "message": "Job started",
        "job_id": job_id,
        "status": "queued"
    }

@app.get("/job-status/{job_id}")
def job_status(job_id: str):

    if job_id not in JOBS:
        return {"error": "Job not found"}

    return JOBS[job_id]

@app.get("/job-results/{job_id}")
def job_results(job_id: str, sort_by: str = "timeline"):

    if job_id not in JOBS:
        return {"error": "Job not found"}

    job = JOBS[job_id]

    if job["status"] != "complete":
        return {
            "job_id": job_id,
            "status": job["status"],
            "stage": job.get("stage", "Processing..."),
            "progress": job.get("progress", 0),
            "error": job.get("error"),
            "message": "Job is not complete yet"
        }

    clips = job.get("clips", [])

    if sort_by == "score":
        clips = sorted(clips, key=lambda x: x.get("score", 0), reverse=True)
    else:
        clips = sorted(clips, key=lambda x: x["start"])

    cleaned_clips = []

    for clip in clips:
        cleaned_clips.append({
            "clip_number": clip["clip_number"],
            "start": clip["start"],
            "end": clip["end"],
            "duration": clip["duration"],
            "text": clip["text"],
            "score": clip.get("score"),
            "output_file": clip["output_file"],
            "thumbnail_file": clip.get("thumbnail_file"),
            "thumbnail_url": f"/view-thumbnail/{job_id}/{clip['clip_number']}",
            "view_url": f"/view-clip/{job_id}/{clip['clip_number']}"
        })
        
     

    return {
        "job_id": job_id,
        "status": job["status"],
        "stage": job.get("stage", "Ready"),
        "progress": job.get("progress", 100),
        "upload_info": job.get("upload_info"),
        "video_duration": job.get("video_duration"),
        "audio_duration": job.get("audio_duration"),
        "min_clip_duration": job.get("min_clip_duration"),
        "max_clip_duration": job.get("max_clip_duration"),
        "raw_clip_candidates": job.get("raw_clip_candidates"),
        "filtered_clip_candidates": job.get("filtered_clip_candidates"),
        "clip_count": job.get("clip_count", 0),
        "timings": job.get("timings"),
        "clips": cleaned_clips
    }

@app.get("/view-clip/{job_id}/{clip_number}")
def view_clip(job_id: str, clip_number: int):

    if job_id not in JOBS:
        return {"error": "Job not found"}

    job = JOBS[job_id]

    if job["status"] != "complete":
        return {"error": "Job is not complete yet"}

    clips = job.get("clips", [])

    matching_clip = None
    for clip in clips:
        if clip["clip_number"] == clip_number:
            matching_clip = clip
            break

    if matching_clip is None:
        return {"error": "Clip not found"}

    file_path = matching_clip["output_file"]

    if not os.path.exists(file_path):
        return {"error": "Clip file does not exist on disk"}

    return FileResponse(file_path, media_type="video/mp4", filename=os.path.basename(file_path))

@app.get("/view-thumbnail/{job_id}/{clip_number}")
def view_thumbnail(job_id: str, clip_number: int):
    file_path = os.path.join(THUMBNAILS_PATH, f"{job_id}_clip_{clip_number}.jpg")

    if not os.path.exists(file_path):
        return {"error": "Thumbnail not found"}

    return FileResponse(file_path, media_type="image/jpeg", filename=os.path.basename(file_path))

@app.get("/download-all/{job_id}")
def download_all(job_id: str):

    if job_id not in JOBS:
        return {"error": "Job not found"}

    job = JOBS[job_id]

    if job["status"] != "complete":
        return {"error": "Job not complete"}

    clips = job.get("clips", [])
    

    if not clips:
        return {"error": "No clips found"}

    zip_path = os.path.join(FINAL_CLIPS_PATH, f"{job_id}_all_clips.zip")

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for clip in clips:
            file_path = os.path.join(
                FINAL_CLIPS_PATH,
                f"{job_id}_clip_{clip['clip_number']}_final.mp4"
            )

            if os.path.exists(file_path):
                zipf.write(file_path, os.path.basename(file_path))

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path)
    )

@app.get("/view-final/{job_id}/{clip_number}")
def view_final(job_id: str, clip_number: int):

    file_path = os.path.join(FINAL_CLIPS_PATH, f"{job_id}_clip_{clip_number}_final.mp4")

    if not os.path.exists(file_path):
        return {"error": "Final clip not found"}

    return FileResponse(file_path, media_type="video/mp4", filename=os.path.basename(file_path))

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
    

def run_final_render_job(render_job_id, job_id, clip_number):
    try:
        start_total_render = time.time()
        render_timings = {}

        RENDER_JOBS[render_job_id]["status"] = "processing"
        RENDER_JOBS[render_job_id]["stage"] = "Preparing render..."
        RENDER_JOBS[render_job_id]["progress"] = 10

        if job_id not in JOBS:
            RENDER_JOBS[render_job_id]["status"] = "failed"
            RENDER_JOBS[render_job_id]["stage"] = "Render failed"
            RENDER_JOBS[render_job_id]["progress"] = 0
            RENDER_JOBS[render_job_id]["error"] = "Job not found"
            return

        job = JOBS[job_id]

        if job["status"] != "complete":
            RENDER_JOBS[render_job_id]["status"] = "failed"
            RENDER_JOBS[render_job_id]["stage"] = "Render failed"
            RENDER_JOBS[render_job_id]["progress"] = 0
            RENDER_JOBS[render_job_id]["error"] = "Job is not complete yet"
            return

        matching_clip = None
        for clip in job.get("clips", []):
            if clip["clip_number"] == clip_number:
                matching_clip = clip
                break

        if matching_clip is None:
            RENDER_JOBS[render_job_id]["status"] = "failed"
            RENDER_JOBS[render_job_id]["stage"] = "Render failed"
            RENDER_JOBS[render_job_id]["progress"] = 0
            RENDER_JOBS[render_job_id]["error"] = "Clip not found"
            return

        os.makedirs(CAPTIONS_PATH, exist_ok=True)
        os.makedirs(FINAL_CLIPS_PATH, exist_ok=True)

        input_clip = matching_clip["output_file"]
        caption_file = os.path.join(CAPTIONS_PATH, f"{job_id}_clip_{clip_number}.srt")
        vertical_file = os.path.join(FINAL_CLIPS_PATH, f"{job_id}_clip_{clip_number}_vertical.mp4")
        final_file = os.path.join(FINAL_CLIPS_PATH, f"{job_id}_clip_{clip_number}_final.mp4")

        RENDER_JOBS[render_job_id]["stage"] = "Creating captions..."
        RENDER_JOBS[render_job_id]["progress"] = 30
        start_caption_creation = time.time()        

        def format_srt_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

        clip_start = matching_clip["start"]
        clip_end = matching_clip["end"]

        with open(SEGMENTS_PATH, "r", encoding="utf-8") as f:
            all_segments = json.load(f)

        clip_segments = []
        for seg in all_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]

            # keep segments that overlap this clip
            if seg_end > clip_start and seg_start < clip_end:
                relative_start = max(seg_start, clip_start) - clip_start
                relative_end = min(seg_end, clip_end) - clip_start
                segment_text = seg["text"].strip()

                if segment_text:
                    clip_segments.append({
                        "start": relative_start,
                        "end": relative_end,
                        "text": segment_text
                    })

        with open(caption_file, "w", encoding="utf-8") as f:
            for index, seg in enumerate(clip_segments, start=1):
                f.write(f"{index}\n")
                f.write(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n")
                f.write(f"{seg['text']}\n\n")

        render_timings["caption_creation"] = round(time.time() - start_caption_creation, 2)

        RENDER_JOBS[render_job_id]["stage"] = "Formatting vertical video..."
        RENDER_JOBS[render_job_id]["progress"] = 55
        start_vertical_render = time.time()

        crop_filter = build_face_centered_crop_filter(input_clip)

        vertical_command = [
            "ffmpeg",
            "-y",
            "-i", input_clip,
            "-vf", crop_filter,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-c:a", "aac",
            vertical_file
        ]        

        vertical_result = subprocess.run(vertical_command, capture_output=True, text=True)
        render_timings["vertical_render"] = round(time.time() - start_vertical_render, 2)        

        if vertical_result.returncode != 0:
            RENDER_JOBS[render_job_id]["status"] = "failed"
            RENDER_JOBS[render_job_id]["stage"] = "Render failed"
            RENDER_JOBS[render_job_id]["progress"] = 0
            RENDER_JOBS[render_job_id]["error"] = "Vertical render failed"
            RENDER_JOBS[render_job_id]["stderr"] = vertical_result.stderr[-1000:]
            return

        subtitle_file_ffmpeg = caption_file.replace("\\", "/").replace(":", "\\:")

        RENDER_JOBS[render_job_id]["stage"] = "Burning captions..."
        RENDER_JOBS[render_job_id]["progress"] = 80
        start_caption_burn = time.time()

        burn_command = [
            "ffmpeg",
            "-y",
            "-i", vertical_file,
            "-vf", f"subtitles='{subtitle_file_ffmpeg}':force_style='FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=75'",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "30",
            "-c:a", "aac",
            final_file
        ]

        burn_result = subprocess.run(burn_command, capture_output=True, text=True)
        render_timings["caption_burn"] = round(time.time() - start_caption_burn, 2)        

        if burn_result.returncode != 0:
            RENDER_JOBS[render_job_id]["status"] = "failed"
            RENDER_JOBS[render_job_id]["stage"] = "Render failed"
            RENDER_JOBS[render_job_id]["progress"] = 0
            RENDER_JOBS[render_job_id]["error"] = "Caption burn failed"
            RENDER_JOBS[render_job_id]["stderr"] = burn_result.stderr[-1000:]
            return

        render_timings["total"] = round(time.time() - start_total_render, 2)

        RENDER_JOBS[render_job_id]["status"] = "complete"
        RENDER_JOBS[render_job_id]["stage"] = "Ready"
        RENDER_JOBS[render_job_id]["progress"] = 100
        RENDER_JOBS[render_job_id]["job_id"] = job_id
        RENDER_JOBS[render_job_id]["clip_number"] = clip_number
        RENDER_JOBS[render_job_id]["final_file"] = final_file
        RENDER_JOBS[render_job_id]["timings"] = render_timings
        
    except Exception as e:
        RENDER_JOBS[render_job_id]["status"] = "failed"
        RENDER_JOBS[render_job_id]["stage"] = "Render failed"
        RENDER_JOBS[render_job_id]["progress"] = 0
        RENDER_JOBS[render_job_id]["error"] = str(e)      


@app.post("/start-final-render/{job_id}/{clip_number}")
def start_final_render(job_id: str, clip_number: int):
    render_job_id = str(uuid.uuid4())

    RENDER_JOBS[render_job_id] = {
        "status": "queued",
        "stage": "Queued",
        "progress": 5,
        "job_id": job_id,
        "clip_number": clip_number
    }

    thread = threading.Thread(
        target=run_final_render_job,
        args=(render_job_id, job_id, clip_number)
    )
    thread.start()

    return {
        "message": "Final render started",
        "render_job_id": render_job_id,
        "status": "queued"
    }

@app.get("/final-render-status/{render_job_id}")
def final_render_status(render_job_id: str):
    if render_job_id not in RENDER_JOBS:
        return {"error": "Render job not found"}

    return RENDER_JOBS[render_job_id]