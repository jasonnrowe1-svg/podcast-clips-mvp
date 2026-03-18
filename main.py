from fastapi.responses import HTMLResponse
from fastapi import FastAPI
import subprocess
import json
import os
import whisper
from fastapi import UploadFile, File
import shutil
import threading
import uuid
from fastapi.responses import FileResponse
import zipfile

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

    global CURRENT_VIDEO

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join("uploads", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    CURRENT_VIDEO = file_path

    return {
        "message": "Video uploaded",
        "file_path": file_path
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


def score_clip_text(text, duration):
    score = 0
    cleaned = text.strip()
    lower_text = cleaned.lower()

    word_count = len(cleaned.split())
    score += word_count

    if duration > 0:
        word_density = word_count / duration
        score += word_density * 10

    if cleaned.endswith((".", "!", "?")):
        score += 5
    else:
        score -= 3

    filler_starts = ["um", "uh", "so", "well", "like", "you know"]
    for filler in filler_starts:
        if lower_text.startswith(filler):
            score -= 5
            break

    weak_endings = ["and", "but", "so", "because", "then", "or", "if", "to"]
    words = cleaned.split()
    if words:
        last_word = words[-1].lower().rstrip(".,!?")
        if last_word in weak_endings:
            score -= 6

    if word_count < 8:
        score -= 4

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

    weak_starts = ["and", "but", "so", "because", "then", "or", "if", "well"]
    weak_endings = ["and", "but", "so", "because", "then", "or", "if", "to"]

    if first_word in weak_starts:
        return False

    if last_word in weak_endings:
        return False

    if not cleaned.endswith((".", "!", "?")):
        return False

    return True


def run_pipeline_job(job_id, video_path):
    try:
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["stage"] = "extracting audio"

        os.makedirs("output", exist_ok=True)
        os.makedirs("transcripts", exist_ok=True)
        os.makedirs(CLIPS_PATH, exist_ok=True)
        os.makedirs(THUMBNAILS_PATH, exist_ok=True)

        # extract audio
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

        if extract_result.returncode != 0:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = "Audio extraction failed"
            return
        
        audio_duration = get_audio_duration(AUDIO_PATH)
        JOBS[job_id]["audio_duration"] = audio_duration

        if audio_duration < 1:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["stage"] = "failed"
            JOBS[job_id]["error"] = "Extracted audio is too short or empty"
            return
        

        JOBS[job_id]["stage"] = "transcribing"


        # transcribe
        result = model.transcribe(AUDIO_PATH)

        with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
            f.write(result["text"])

        with open(SEGMENTS_PATH, "w", encoding="utf-8") as f:
            json.dump(result["segments"], f, indent=2)

        video_duration = get_video_duration(video_path)
        min_clip_duration, max_clip_duration = get_clip_length_targets(video_duration)

        JOBS[job_id]["video_duration"] = video_duration
        JOBS[job_id]["min_clip_duration"] = min_clip_duration
        JOBS[job_id]["max_clip_duration"] = max_clip_duration


        JOBS[job_id]["stage"] = "building clip candidates"


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

                    if not has_good_clip_boundaries(clip_text):
                        continue

                    clip_score = score_clip_text(clip_text, duration)

                    clips.append({
                        "start": start_time,
                        "end": end_time,
                        "duration": round(duration, 2),
                        "text": clip_text,
                        "score": clip_score
                    })
                    break
        JOBS[job_id]["stage"] = "ranking clips"
        clips.sort(key=lambda x: x["score"], reverse=True)
        JOBS[job_id]["raw_clip_candidates"] = len(clips)
        
        MAX_CLIPS = 8
        clips = clips[:MAX_CLIPS]

        # keep best non-overlapping clips
        filtered_clips = []
        remaining_clips = clips.copy()

        while remaining_clips:
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
        JOBS[job_id]["stage"] = "cutting clips"
        created_clips = []

        for index, clip in enumerate(clips, start=1):
            output_file = os.path.join(CLIPS_PATH, f"{job_id}_clip_{index}.mp4")
            thumbnail_file = os.path.join(THUMBNAILS_PATH, f"{job_id}_clip_{index}.jpg")
            
            JOBS[job_id]["stage"] = f"cutting clip {index}/{len(clips)}"

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
            
            JOBS[job_id]["stage"] = f"generating thumbnails ({index}/{len(clips)})"

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

        JOBS[job_id]["status"] = "complete"
        JOBS[job_id]["stage"] = "complete"
        JOBS[job_id]["segment_count"] = len(result["segments"])
        JOBS[job_id]["clip_count"] = len(created_clips)
        JOBS[job_id]["clips"] = created_clips

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["stage"] = "failed"
        JOBS[job_id]["error"] = str(e)

@app.post("/start-job")
def start_job():

    if CURRENT_VIDEO is None:
        return {"error": "No video uploaded yet"}

    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "status": "queued",
        "stage": "queued",
        "video": CURRENT_VIDEO
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
            "stage": job.get("stage"),
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
        "stage": job.get("stage"),
        "video_duration": job.get("video_duration"),
        "audio_duration": job.get("audio_duration"),
        "min_clip_duration": job.get("min_clip_duration"),
        "max_clip_duration": job.get("max_clip_duration"),
        "raw_clip_candidates": job.get("raw_clip_candidates"),
        "filtered_clip_candidates": job.get("filtered_clip_candidates"),
        "clip_count": job.get("clip_count", 0),
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
        RENDER_JOBS[render_job_id]["status"] = "processing"
        RENDER_JOBS[render_job_id]["stage"] = "preparing"

        if job_id not in JOBS:
            RENDER_JOBS[render_job_id]["status"] = "failed"
            RENDER_JOBS[render_job_id]["stage"] = "failed"
            RENDER_JOBS[render_job_id]["error"] = "Job not found"
            return

        job = JOBS[job_id]

        if job["status"] != "complete":
            RENDER_JOBS[render_job_id]["status"] = "failed"
            RENDER_JOBS[render_job_id]["stage"] = "failed"
            RENDER_JOBS[render_job_id]["error"] = "Job is not complete yet"
            return

        matching_clip = None
        for clip in job.get("clips", []):
            if clip["clip_number"] == clip_number:
                matching_clip = clip
                break

        if matching_clip is None:
            RENDER_JOBS[render_job_id]["status"] = "failed"
            RENDER_JOBS[render_job_id]["stage"] = "failed"
            RENDER_JOBS[render_job_id]["error"] = "Clip not found"
            return

        os.makedirs(CAPTIONS_PATH, exist_ok=True)
        os.makedirs(FINAL_CLIPS_PATH, exist_ok=True)

        input_clip = matching_clip["output_file"]
        caption_file = os.path.join(CAPTIONS_PATH, f"{job_id}_clip_{clip_number}.srt")
        vertical_file = os.path.join(FINAL_CLIPS_PATH, f"{job_id}_clip_{clip_number}_vertical.mp4")
        final_file = os.path.join(FINAL_CLIPS_PATH, f"{job_id}_clip_{clip_number}_final.mp4")

        RENDER_JOBS[render_job_id]["stage"] = "creating captions"

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

        RENDER_JOBS[render_job_id]["stage"] = "making vertical version"

        vertical_command = [
            "ffmpeg",
            "-y",
            "-i", input_clip,
            "-vf",
            "scale=720:1280:force_original_aspect_ratio=increase,"
            "crop=720:1280",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-c:a", "aac",
            vertical_file
        ]

        vertical_result = subprocess.run(vertical_command, capture_output=True, text=True)

        if vertical_result.returncode != 0:
            RENDER_JOBS[render_job_id]["status"] = "failed"
            RENDER_JOBS[render_job_id]["stage"] = "failed"
            RENDER_JOBS[render_job_id]["error"] = "Vertical render failed"
            RENDER_JOBS[render_job_id]["stderr"] = vertical_result.stderr[-1000:]
            return

        subtitle_file_ffmpeg = caption_file.replace("\\", "/").replace(":", "\\:")

        RENDER_JOBS[render_job_id]["stage"] = "burning captions (fast mode)"

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

        if burn_result.returncode != 0:
            RENDER_JOBS[render_job_id]["status"] = "failed"
            RENDER_JOBS[render_job_id]["stage"] = "failed"
            RENDER_JOBS[render_job_id]["error"] = "Caption burn failed"
            RENDER_JOBS[render_job_id]["stderr"] = burn_result.stderr[-1000:]
            return

        RENDER_JOBS[render_job_id]["status"] = "complete"
        RENDER_JOBS[render_job_id]["stage"] = "complete"
        RENDER_JOBS[render_job_id]["job_id"] = job_id
        RENDER_JOBS[render_job_id]["clip_number"] = clip_number
        RENDER_JOBS[render_job_id]["final_file"] = final_file
        
    except Exception as e:
        RENDER_JOBS[render_job_id]["status"] = "failed"
        RENDER_JOBS[render_job_id]["stage"] = "failed"
        RENDER_JOBS[render_job_id]["error"] = str(e)      


@app.post("/start-final-render/{job_id}/{clip_number}")
def start_final_render(job_id: str, clip_number: int):
    render_job_id = str(uuid.uuid4())

    RENDER_JOBS[render_job_id] = {
        "status": "queued",
        "stage": "queued",
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

@app.post("/start-final-render/{job_id}/{clip_number}")
def start_final_render(job_id: str, clip_number: int):
    render_job_id = str(uuid.uuid4())

    RENDER_JOBS[render_job_id] = {
        "status": "queued",
        "stage": "queued",
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