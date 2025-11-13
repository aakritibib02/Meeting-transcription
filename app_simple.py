"""
Meeting Transcription Backend Server - Simplified Version
========================================================

A Flask-based backend server that provides:
- Real-time speech-to-text transcription using OpenAI Whisper
- Multi-language translation using Transformers
- Task detection and assignment using NLP
- Meeting summary generation
- Email integration for task assignments

Simplified version without complex audio processing dependencies.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Flask and CORS
from flask import Flask, request, jsonify
from flask_cors import CORS

# Audio and AI processing
import numpy as np
import base64

# AI Libraries
import whisper
from transformers import pipeline
import spacy
from sentence_transformers import SentenceTransformer
import torch

# Utilities
import threading
import time
from collections import defaultdict, deque
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeetingTranscriptionBackend:
    """Simplified backend class handling all AI processing"""
    
    def __init__(self):
        self.app = Flask(__name__)
        
        # Configure CORS for Chrome extension and meeting platforms
        CORS(self.app, resources={
            r"/*": {
                "origins": ["*"],
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
                "supports_credentials": True,
                "max_age": 3600
            }
        })
        
        # Configuration
        self.config = self.load_config()
        
        # AI Models (will be loaded lazily)
        self.whisper_model = None
        self.translation_models = {}
        self.nlp_model = None
        self.sentence_model = None
        
        # Meeting data storage
        self.meetings = {}
        self.current_meeting = None
        
        self.setup_routes()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "whisper_model": "base",  # tiny, base, small, medium, large
            "max_audio_length": 30,   # seconds
            "translation_cache_size": 1000,
            "task_keywords": [
                "need to", "have to", "must", "should", "will", "going to",
                "action item", "task", "todo", "assignment", "follow up",
                "deadline", "due date", "by when", "responsible for"
            ],
            "priority_keywords": {
                "high": ["urgent", "asap", "immediately", "critical", "high priority"],
                "low": ["eventually", "when possible", "low priority", "nice to have"]
            },
            "supported_languages": {
                "en": "English", "es": "Spanish", "fr": "French", "de": "German",
                "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
                "ko": "Korean", "zh": "Chinese"
            }
        }
        
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config: {e}, using defaults")
                
        return default_config
    
    def initialize_models(self):
        """Initialize AI models lazily"""
        logger.info("Initializing AI models...")
        
        try:
            # Initialize Whisper for speech recognition
            if self.whisper_model is None:
                logger.info(f"Loading Whisper model: {self.config['whisper_model']}")
                self.whisper_model = whisper.load_model(self.config['whisper_model'])
                logger.info("Whisper model loaded successfully")
            
            # Initialize spaCy for NLP tasks
            if self.nlp_model is None:
                try:
                    self.nlp_model = spacy.load("en_core_web_sm")
                    logger.info("spaCy model loaded successfully")
                except OSError:
                    logger.warning("spaCy model not found")
                    self.nlp_model = None
            
            # Initialize sentence transformer for semantic analysis
            if self.sentence_model is None:
                logger.info("Loading sentence transformer model...")
                try:
                    self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Sentence transformer loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load sentence transformer: {e}")
                    self.sentence_model = None
                
            logger.info("AI models initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def setup_routes(self):
        """Setup Flask routes for the API"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "models_loaded": {
                    "whisper": self.whisper_model is not None,
                    "nlp": self.nlp_model is not None,
                    "sentence_transformer": self.sentence_model is not None
                }
            })
        
        @self.app.route('/start_meeting', methods=['POST'])
        def start_meeting():
            """Start a new meeting session"""
            try:
                data = request.json or {}
                meeting_id = data.get('meeting_id', f"meeting_{int(time.time())}")
                settings = data.get('settings', {})
                
                # Initialize models if not already done
                if self.whisper_model is None:
                    self.initialize_models()
                
                # Create new meeting session
                self.current_meeting = meeting_id
                self.meetings[meeting_id] = {
                    "id": meeting_id,
                    "start_time": datetime.now(),
                    "settings": settings,
                    "transcriptions": [],
                    "translations": [],
                    "tasks": [],
                    "summary": "",
                    "participants": []
                }
                
                return jsonify({
                    "success": True,
                    "meeting_id": meeting_id,
                    "message": "Meeting started successfully"
                })
                
            except Exception as e:
                logger.error(f"Error starting meeting: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/transcribe_audio', methods=['POST'])
        def transcribe_audio():
            """Transcribe audio chunk (simplified - expects text input for demo)"""
            try:
                data = request.json or {}
                # For demo purposes, accept text directly
                text_input = data.get('text')
                if text_input:
                    transcription = {
                        "text": text_input,
                        "language": "en",
                        "confidence": 0.9,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Store transcription
                    if self.current_meeting:
                        meeting = self.meetings[self.current_meeting]
                        meeting["transcriptions"].append(transcription)
                    
                    return jsonify({
                        "success": True,
                        "transcription": transcription
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "No text provided for transcription demo"
                    })
                    
            except Exception as e:
                logger.error(f"Error transcribing: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/translate_text', methods=['POST'])
        def translate_text():
            """Translate text to target language using free Helsinki-NLP models (CPU)."""
            try:
                data = request.json or {}
                text = data.get('text')
                source_lang = (data.get('source_lang') or 'en').split('-')[0].lower()
                target_lang = (data.get('target_lang') or 'en').split('-')[0].lower()
                
                if not text:
                    return jsonify({"success": False, "error": "No text provided"})
                
                # Normalize a few codes
                alias = { 'zh-cn': 'zh', 'zh-tw': 'zh', 'pt-br': 'pt', 'en-us': 'en', 'en-gb': 'en' }
                source_lang = alias.get(source_lang, source_lang)
                target_lang = alias.get(target_lang, target_lang)
                
                translation = self.translate_text_hf(text, source_lang, target_lang)
                
                # Store translation
                if self.current_meeting:
                    meeting = self.meetings[self.current_meeting]
                    meeting["translations"].append({
                        "timestamp": datetime.now().isoformat(),
                        "original_text": text,
                        "translated_text": translation,
                        "source_language": source_lang,
                        "target_language": target_lang
                    })
                
                return jsonify({
                    "success": True,
                    "translation": translation
                })
                
            except Exception as e:
                logger.error(f"Error translating text: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

    def translate_text_hf(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using HuggingFace transformers with Helsinki-NLP MarianMT models.
        Optimized for English -> target flows; falls back gracefully if model unavailable.
        """
        try:
            if not text.strip():
                return text
            
            # If target is English, return as-is (you can change to real translation if needed)
            if target_lang == 'en':
                return text
            
            # For simplicity, assume English source if auto/unknown
            if source_lang in ('auto', 'unknown', None):
                source_lang = 'en'
            
            # Prefer direct en->target models (common use case)
            if source_lang == 'en':
                model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
                cache_key = f"en-{target_lang}"
                
                if cache_key not in self.translation_models:
                    try:
                        self.translation_models[cache_key] = pipeline(
                            "translation",
                            model=model_name,
                            device=0 if torch.cuda.is_available() else -1
                        )
                    except Exception as e:
                        logger.warning(f"Could not load model {model_name}: {e}")
                        return f"[Translated to {target_lang}] {text}"
                
                translator = self.translation_models[cache_key]
                out = translator(text, max_length=512)
                return out[0]['translation_text']
            
            # Fallback: if non-English source requested, try source->target directly
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            cache_key = f"{source_lang}-{target_lang}"
            if cache_key not in self.translation_models:
                try:
                    self.translation_models[cache_key] = pipeline(
                        "translation",
                        model=model_name,
                        device=0 if torch.cuda.is_available() else -1
                    )
                except Exception as e:
                    logger.warning(f"Could not load model {model_name}: {e}")
                    return f"[Translated to {target_lang}] {text}"
            translator = self.translation_models[cache_key]
            out = translator(text, max_length=512)
            return out[0]['translation_text']
        except Exception as e:
            logger.error(f"Translation failure: {e}")
            return f"[Translation Error] {text}"
        
        @self.app.route('/detect_tasks', methods=['POST'])
        def detect_tasks():
            """Detect tasks and action items from text"""
            try:
                data = request.json or {}
                text = data.get('text')
                
                if not text:
                    return jsonify({"success": False, "error": "No text provided"})
                
                tasks = self.extract_tasks_from_text(text)
                
                # Store tasks
                if self.current_meeting and tasks:
                    meeting = self.meetings[self.current_meeting]
                    meeting["tasks"].extend(tasks)
                
                return jsonify({
                    "success": True,
                    "tasks": tasks
                })
                
            except Exception as e:
                logger.error(f"Error detecting tasks: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/generate_summary', methods=['POST'])
        def generate_summary():
            """Generate meeting summary"""
            try:
                data = request.json or {}
                meeting_id = data.get('meeting_id', self.current_meeting)
                
                if not meeting_id or meeting_id not in self.meetings:
                    return jsonify({"success": False, "error": "Meeting not found"})
                
                meeting = self.meetings[meeting_id]
                summary = self.generate_meeting_summary(meeting)
                meeting["summary"] = summary
                
                return jsonify({
                    "success": True,
                    "summary": summary
                })
                
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
    
    def extract_tasks_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tasks using pattern matching"""
        tasks = []
        
        try:
            # Task detection patterns
            task_patterns = [
                r"(?:need to|have to|must|should|will|going to)\s+([^.!?]+)",
                r"(?:action item|task|todo|assignment):\s*([^.!?]+)",
                r"(?:@\w+|assign|assigned to|responsible for)\s+([^.!?]+)",
                r"(?:by|due|deadline|before)\s+([^.!?]+)"
            ]
            
            potential_tasks = []
            for pattern in task_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    task_text = match.group(1).strip()
                    if len(task_text) > 3:
                        potential_tasks.append({
                            "text": task_text,
                            "start": match.start(),
                            "pattern": pattern
                        })
            
            # Process each potential task
            for i, potential_task in enumerate(potential_tasks):
                task_text = potential_task["text"]
                
                # Extract assignee
                assignee = None
                mention_match = re.search(r"@(\w+)", text)
                if mention_match:
                    assignee = mention_match.group(1)
                
                # Determine priority
                priority = self.determine_task_priority(text, potential_task["start"])
                
                # Create task object
                task = {
                    "id": f"task_{int(time.time())}_{i}",
                    "description": task_text,
                    "assignee": assignee,
                    "priority": priority,
                    "timestamp": datetime.now().isoformat(),
                    "status": "pending",
                    "confidence": 0.8
                }
                
                tasks.append(task)
            
        except Exception as e:
            logger.error(f"Error extracting tasks: {e}")
        
        return tasks
    
    def determine_task_priority(self, text: str, position: int) -> str:
        """Determine task priority from context"""
        context_start = max(0, position - 100)
        context_end = min(len(text), position + 100)
        context = text[context_start:context_end].lower()
        
        # Check for high priority indicators
        high_priority_words = self.config["priority_keywords"]["high"]
        if any(word in context for word in high_priority_words):
            return "high"
        
        # Check for low priority indicators
        low_priority_words = self.config["priority_keywords"]["low"]
        if any(word in context for word in low_priority_words):
            return "low"
        
        return "normal"
    
    def generate_meeting_summary(self, meeting: Dict[str, Any]) -> str:
        """Generate meeting summary"""
        try:
            summary_parts = []
            
            # Header
            start_time = meeting.get("start_time", datetime.now())
            end_time = meeting.get("end_time", datetime.now())
            
            summary_parts.append("MEETING SUMMARY")
            summary_parts.append("=" * 50)
            summary_parts.append(f"Meeting ID: {meeting['id']}")
            summary_parts.append(f"Date: {start_time.strftime('%Y-%m-%d')}")
            summary_parts.append(f"Start Time: {start_time.strftime('%H:%M:%S')}")
            summary_parts.append("")
            
            # Transcriptions
            transcriptions = meeting.get("transcriptions", [])
            if transcriptions:
                summary_parts.append("DISCUSSION POINTS:")
                summary_parts.append("-" * 20)
                for i, transcript in enumerate(transcriptions[-5:], 1):  # Last 5 items
                    summary_parts.append(f"{i}. {transcript['text']}")
                summary_parts.append("")
            
            # Tasks
            tasks = meeting.get("tasks", [])
            if tasks:
                summary_parts.append("ACTION ITEMS:")
                summary_parts.append("-" * 15)
                for i, task in enumerate(tasks, 1):
                    summary_parts.append(f"{i}. {task['description']}")
                    if task.get("assignee"):
                        summary_parts.append(f"   Assigned to: {task['assignee']}")
                    summary_parts.append(f"   Priority: {task.get('priority', 'normal')}")
                    summary_parts.append("")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating meeting summary."
    
    def run(self, host='localhost', port=5000, debug=False):
        """Run the Flask server"""
        logger.info(f"Starting Meeting Transcription Backend on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def main():
    """Main entry point"""
    backend = MeetingTranscriptionBackend()
    
    try:
        backend.run(host='localhost', port=5000, debug=False)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()