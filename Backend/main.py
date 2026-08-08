import os
import json
import time
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from Backend.meme import generate_meme_content, add_meme_text, generate_meme_image
from Backend.blog import fetch_news_newsapi, fetch_arxiv, fetch_wikipedia, generate_blog_with_gemini, duckduckgo_instant_answer, save_json_to_file, generate_image_from_prompt, generate_images_from_blog_json, save_markdown_file, read_markdown_content
from Backend.images import generate_image_from_topic_and_style
from Backend.text_generator import generate_video_script, extract_json, save_json
from Backend.image_generator import generate_images_from_script
from Backend.audio_generator import generate_audio
from Backend.music_generator import generate_music, generate_music_prompt
from Backend.video_generator import create_final_video
from Backend.logger import logging  
from Backend.chatbot import KnowledgeBase
from Backend.db_utils import DatabaseManager


# FastAPI setup
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for input validation
class MemeRequest(BaseModel):
    topic: str
    style: str
    language: str

class BlogRequest(BaseModel):
    topic: str
    style: str
    length: str

class ImageRequest(BaseModel):
    topic: str
    style: str

class VideoRequest(BaseModel):
    topic: str
    style: str
    
class ChatRequest(BaseModel):
    query: str

class SetupConfig(BaseModel):
    mongo_uri: str
    gemini_key: str
    huggingface_key: str

    @validator('mongo_uri')
    def validate_mongo_uri(cls, v):
        v = v.strip().strip("'\"")
        if not v.startswith('mongodb'):
            raise ValueError('Invalid MongoDB URI')
        return v

    @validator('gemini_key')
    def validate_gemini_key(cls, v):
        v = v.strip().strip("'\"")
        if len(v) < 15:  
            raise ValueError('Invalid Gemini API key')
        return v

    @validator('huggingface_key')
    def validate_huggingface_key(cls, v):
        v = v.strip().strip("'\"")
        if len(v) < 15: 
            raise ValueError('Invalid HuggingFace API key')
        return v
    
kb = KnowledgeBase()

@app.get("/", response_class=FileResponse)
async def serve_home():
    logging.info("Home page served.")
    return FileResponse(BASE_DIR/"static/home.html")

@app.post("/generate_meme")
async def generate_meme(request: MemeRequest):
    logging.info("Meme generation request received.")
    try:
        topic, style, language = request.topic, request.style, request.language
        meme_caption, image_prompt, text_color, text_position, language = generate_meme_content(topic, style, language)

        meme_image = generate_meme_image(image_prompt)
        # meme_image_path = os.path.join("static", "generated_meme.png")
        current_time = str(int(time.time()))  # Get current time in seconds (integer)
        unique_filename = f"generated_meme_{current_time}.png"

        meme_image_path = os.path.join("static/memes", unique_filename)
        os.makedirs(os.path.dirname(meme_image_path), exist_ok=True)
        meme_image.save(meme_image_path)

        final_meme = add_meme_text(meme_image, meme_caption, text_color, text_position, language)
        final_meme.save(meme_image_path)

        logging.info("Meme successfully generated.")
        return {"caption": meme_caption, "image_url": f"/static/memes/{unique_filename}"}

    except Exception as e:
        logging.error(f"Error generating meme: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating meme: {str(e)}")

@app.get("/static/{file_path}")
async def get_file(file_path: str):
    file_location = os.path.join("static", file_path)
    if os.path.exists(file_location):
        logging.info(f"Serving static file: {file_path}")
        return FileResponse(file_location)
    else:
        logging.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

@app.post("/generate_blog")
async def generate_blog(request: BlogRequest):
    logging.info("Blog generation request received.")
    topic, style, length = request.topic, request.style, request.length
    try:
        context = ""
        sources = [fetch_wikipedia, fetch_arxiv, fetch_news_newsapi, duckduckgo_instant_answer]
        for source in sources:
            context += source(topic) + "\n\n"

        raw_blog = generate_blog_with_gemini(context, topic, style, length)
        if isinstance(raw_blog, dict) and "error" in raw_blog:
            raise ValueError(raw_blog["error"])

        parsed_data = save_json_to_file(raw_blog, filename="static/blogs/final_blog.json")
        if parsed_data is None:
            raise ValueError("Invalid blog JSON structure.")

        image_paths = generate_images_from_blog_json(parsed_data)
        markdown_path = save_markdown_file(parsed_data, image_paths, output_dir="static/blogs")
        show = read_markdown_content(markdown_path=markdown_path)

        logging.info("Blog successfully generated.")
        return show

    except Exception as e:
        logging.error(f"Error generating blog: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating blog: {str(e)}")

@app.post("/generate-image/")
async def generate_image_api(request: ImageRequest):
    logging.info(f"Image generation request received for topic: {request.topic}")
    try:
        description, image_base64 = generate_image_from_topic_and_style(request.topic, request.style)
        logging.info(f"Image generated successfully for topic: {request.topic}")
        return {"description": description, "image": image_base64}

    except RuntimeError as e:
        logging.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
current_path = Path(__file__).resolve()
root_path = current_path.parent.parent
static_path = current_path.parent /'static'
output_data_dir = static_path / 'output/video'
@app.post("/generate_video/")
async def generate_video(request: VideoRequest):
    logging.info(f"Video generation request received for topic: {request.topic}")
    try:
        topic, style = request.topic, request.style
        
        unique_folder = output_data_dir / f"video_{int(time.time())}"
        unique_folder.mkdir(parents=True, exist_ok=True)

        images_output_dir = unique_folder / 'output_images'
        audio_output_dir = unique_folder / 'output_audio'
        music_output_dir = unique_folder / 'output_music'
        images_output_dir.mkdir(parents=True, exist_ok=True)
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        music_output_dir.mkdir(parents=True, exist_ok=True)

        response_text = generate_video_script(topic, style)
        video_script = extract_json(response_text)

        save_json(video_script, unique_folder)

        generate_images_from_script(video_script, images_output_dir)
        generate_audio(video_script, audio_output_dir)

        music_prompt = generate_music_prompt(video_script["background_music_prompt"], video_script["scenes"], video_script["overall_video_mood"])
        generate_music(music_prompt, music_output_dir)

        final_video_path = unique_folder / "final_video.mp4"
        create_final_video(video_script, str(images_output_dir), str(audio_output_dir), str(music_output_dir / "background_music.flac"), str(final_video_path))

        logging.info(f"Video successfully generated: {final_video_path}")
        return FileResponse(final_video_path, media_type="video/mp4")

    except Exception as e:
        logging.error(f"Error generating video: {e}")
        return {"error": str(e)}

@app.post("/api/initialize")
async def initialize_kb():
    try:
        success = kb.initialize()
        if success:
            logging.info("Knowledge base initialized successfully")
            return JSONResponse(content={"success": True, "message": "Knowledge base initialized successfully"})
        else:
            logging.error("Failed to initialize knowledge base")
            return JSONResponse(content={"success": False, "message": "Failed to initialize knowledge base"}, status_code=500)
    except Exception as e:
        logging.error(f"Error initializing knowledge base: {str(e)}")
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)

@app.get("/api/status")
async def get_status():
    return JSONResponse(content={
        "status": "complete" if kb.initialized else "not_started", 
        "initialized": kb.initialized
    })

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        if not kb.initialized:
            raise HTTPException(status_code=400, detail="Knowledge base not initialized")
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Empty query")
            
        response = kb.query(request.query)
        
        if not response:
            raise HTTPException(status_code=500, detail="Failed to generate response")
            
        return JSONResponse(content={"response": response})
        
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Shutting down application")
    if hasattr(kb, 'cleanup'):
        kb.cleanup()

@app.post("/api/setup")
async def setup_config(config: SetupConfig):
    logging.info("Received setup configuration request")
    
    try:
        db = DatabaseManager.get_instance()
        
        # Initialize DB connection
        if not db.initialize(config.mongo_uri):
            logging.error("Failed to initialize database connection")
            raise HTTPException(status_code=500, detail="Failed to connect to database")
        
        # Save API keys
        if not db.save_keys(config.gemini_key, config.huggingface_key):
            logging.error("Failed to save API keys")
            raise HTTPException(status_code=500, detail="Failed to save API keys")
        
        logging.info("Successfully saved configuration")
        return {"status": "success", "message": "Configuration saved successfully"}
    
    except Exception as e:
        logging.error(f"Error in setup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/check-keys")
async def check_keys():
    db = DatabaseManager.get_instance()
    return {
        "using_default_keys": db.is_using_default_keys()
    }
    
@app.get("/api/keys-status")
async def api_keys_status():
    db = DatabaseManager.get_instance()
    try:
        keys = db.get_keys()
        if not keys:
            return {"configured": False, "message": "API keys not configured"}
        return {"configured": True}
    except Exception as e:
        return {"configured": False, "message": str(e)}
    
if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")

    logging.info("Starting FastAPI server.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

