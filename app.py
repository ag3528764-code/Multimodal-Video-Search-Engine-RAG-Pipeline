import os
import cv2
import torch
import librosa
import numpy as np
import gradio as gr
from transformers import CLIPProcessor, CLIPModel, WhisperProcessor, WhisperForConditionalGeneration

# Force CPU fallback if CUDA isn't available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Initializing Video-RAG Engine on: {device.upper()}")

# Initialize Models globally
print("📦 Loading Neural Weights (CLIP & Whisper)...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)

# Global in-memory storage arrays
db_timestamps = []
db_video_embeddings = None
db_text_embeddings = None
db_transcripts = []

def clear_database():
    """Wipes old video vectors on a new upload."""
    global db_timestamps, db_video_embeddings, db_text_embeddings, db_transcripts
    db_timestamps = []
    db_video_embeddings = None
    db_text_embeddings = None
    db_transcripts = []

def process_video_pipeline(video_path):
    """Extracts, vectorizes and indexes both visual and acoustic signals from video."""
    global db_timestamps, db_video_embeddings, db_text_embeddings, db_transcripts
    if not video_path:
        return "⚠️ Error: Please drop an MP4 video file first."
    
    clear_database()
    
    # ----------------------------------------------------
    # PHASE A: VISUAL SAMPLING & EMBEDDING EXTRACTION
    # ----------------------------------------------------
    yield "🎞️ Decoding frames and mapping visual space..."
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25 # Fallback assignment
    
    frame_count = 0
    sampled_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Sample exactly 1 frame per second to keep calculations compact
        if frame_count % int(fps) == 0:
            timestamp = frame_count / fps
            # Convert OpenCV BGR format to standard PIL RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames.append((timestamp, rgb_frame))
        frame_count += 1
    cap.release()

    # Vectorize sampled visual frames in small minibatches
    visual_vectors = []
    for timestamp, frame_img in sampled_frames:
        inputs = clip_processor(images=frame_img, return_tensors="pt").to(device)
        with torch.no_grad():
            frame_emb = clip_model.get_image_features(**inputs)
            # Apply L2 Regularization normalizations
            frame_emb = frame_emb / frame_emb.norm(p=2, dim=-1, keepdim=True)
            visual_vectors.append(frame_emb.cpu().numpy().flatten())
            db_timestamps.append(timestamp)

    db_video_embeddings = np.array(visual_vectors)

    # ----------------------------------------------------
    # PHASE B: AUDIO SPEECH-TO-TEXT DICTIONARY INDEXING
    # ----------------------------------------------------
    yield "🎙️ Extracting audio track and transcribing speech segments..."
    try:
        # Load audio from video natively via librosa
        y, sr = librosa.load(video_path, sr=16000, mono=True)
        # Use a simple chunk stride windows sequence (e.g., 10-second blocks)
        chunk_duration = 10 
        samples_per_chunk = chunk_duration * sr
        
        audio_vectors = []
        for i in range(0, len(y), samples_per_chunk):
            chunk = y[i:i + samples_per_chunk]
            if len(chunk) < sr: # Skip empty remnants
                continue
            
            start_sec = i / sr
            end_sec = min((i + samples_per_chunk) / sr, len(y) / sr)
            mid_timestamp = (start_sec + end_sec) / 2
            
            # Extract transcript tokens
            input_features = whisper_processor(chunk, sampling_rate=sr, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                predicted_ids = whisper_model.generate(input_features)
                transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            if transcription.strip():
                # Extract text embedding vector to align later with queries
                text_inputs = clip_processor(text=[transcription], return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    text_emb = clip_model.get_text_features(**text_inputs)
                    text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
                    audio_vectors.append(text_emb.cpu().numpy().flatten())
                
                db_transcripts.append({
                    "timestamp": mid_timestamp,
                    "text": f"[{int(start_sec)}s - {int(end_sec)}s]: {transcription}"
                })
        
        if len(audio_vectors) > 0:
            db_text_embeddings = np.array(audio_vectors)
    except Exception as e:
        print(f"⚠️ Audio Warning (Could be silent video): {str(e)}")
        db_text_embeddings = None

    yield f"✅ Video Indexed Successfully! Processed {len(db_timestamps)} visual frames and {len(db_transcripts)} speech chunks."

def execute_rag_search(query_text):
    """Executes high-speed cosine tensor similarity vectors indexing over database."""
    global db_timestamps, db_video_embeddings, db_text_embeddings, db_transcripts
    
    if not query_text or db_video_embeddings is None:
        return "⚠️ Upload and register a video dataset first before executing lookup scans."

    # Step 1: Compute query textual semantic vector representations
    inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        query_emb = clip_model.get_text_features(**inputs)
        query_emb = (query_emb / query_emb.norm(p=2, dim=-1, keepdim=True)).cpu().numpy().flatten()

    # Step 2: Compute Dot-Product Cosine Alignment Matches over Video database
    video_scores = np.dot(db_video_embeddings, query_emb)
    best_video_idx = np.argmax(video_scores)
    best_video_time = db_timestamps[best_video_idx]
    best_video_score = video_scores[best_video_idx]

    # Step 3: Compute Dot-Product Alignment over Speech database if available
    best_text_time, best_text_score, matched_transcript = 0, 0.0, "No speech matches found."
    if db_text_embeddings is not None and len(db_transcripts) > 0:
        text_scores = np.dot(db_text_embeddings, query_emb)
        best_text_idx = np.argmax(text_scores)
        best_text_time = db_transcripts[best_text_idx]["timestamp"]
        best_text_score = text_scores[best_text_idx]
        matched_transcript = db_transcripts[best_text_idx]["text"]

    # Step 4: Construct response report
    report = f"🔍 **Search Query Results Matrix:**\n\n"
    report += f"🖼️ **Best Visual Match Timestamp:** `{int(best_video_time)}s` (Confidence: {best_video_score:.2f})\n"
    if db_text_embeddings is not None:
        report += f"🗣️ **Best Spoken Audio Match Timestamp:** `{int(best_text_time)}s` (Confidence: {best_text_score:.2f})\n"
        report += f"📝 **Transcript Snapshot:** *\"{matched_transcript}\"*\n\n"
    
    # Intelligently recommend the absolute winner segment
    winner_time = best_video_time if best_video_score >= best_text_score else best_text_time
    report += f"📌 **Recommendation:** Jump your player trackbar right to **`{int(winner_time)} seconds`** to view this context."
    
    return report

# ----------------------------------------------------
# PHASE C: THE GRAPHICAL FRONTEND INTERFACE
# ----------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 Multimodal Video RAG Engine")
    gr.Markdown("Upload an engineering video file, parse its audio/visual layers, and locate any specific topic instantly using natural language queries.")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Step 1: Upload Targeted MP4 Video File")
            process_btn = gr.Button("⚙️ Parse & Index Multimodal Vectors", variant="primary")
            status_output = gr.Textbox(label="Engine Internal Status Updates", interactive=False)
            
        with gr.Column(scale=1):
            query_input = gr.Textbox(label="Step 2: Enter Search Query", placeholder="e.g., 'A diagram showing rocket engines' or 'when did they talk about metrics?'")
            search_btn = gr.Button("🔍 Execute High-Speed RAG Search")
            results_output = gr.Markdown(label="RAG Final Response")

    # Wire component actions
    process_btn.click(fn=process_video_pipeline, inputs=[video_input], outputs=[status_output])
    search_btn.click(fn=execute_rag_search, inputs=[query_input], outputs=[results_output])

if __name__ == "__main__":
    # Launch local serving pipeline engine instances
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)