"""
Meeting Transcription Backend Server
====================================

A Flask-based backend server that provides:
- Real-time speech-to-text transcription using OpenAI Whisper
- Multi-language translation using Transformers
- Task detection and assignment using NLP
- Meeting summary generation
- Email integration for task assignments

All AI processing happens locally with CPU-friendly optimizations.
No GPU required, uses only free and open-source tools.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Flask and CORS
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# Audio processing
import librosa
import numpy as np
import webrtcvad
from io import BytesIO
import base64

# AI Libraries
import whisper
from transformers import pipeline, MarianMTModel, MarianTokenizer
import spacy
from sentence_transformers import SentenceTransformer
import torch

# Configuration and utilities
import threading
import queue
import time
from collections import defaultdict, deque

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeetingTranscriptionBackend:
    """Main backend class handling all AI processing"""
    
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
        
        # Audio processing
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # Most aggressive VAD mode
        self.audio_queue = queue.Queue()
        self.transcription_buffer = deque(maxlen=50)
        
        # AI Models (will be loaded lazily)
        self.whisper_model = None
        self.translation_models = {}
        self.nlp_model = None
        self.sentence_model = None
        
        # Meeting data storage
        self.meetings = {}
        self.current_meeting = None
        
        # Processing threads
        self.audio_processor_thread = None
        self.is_processing = False
        
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
                "en": "English",
                "es": "Spanish", 
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ru": "Russian",
                "ja": "Japanese",
                "ko": "Korean",
                "zh": "Chinese"
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
        """Initialize AI models lazily for better startup performance"""
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
                    logger.warning("spaCy en_core_web_sm not found. Installing...")
                    os.system("python -m spacy download en_core_web_sm")
                    self.nlp_model = spacy.load("en_core_web_sm")
            
            # Initialize sentence transformer for semantic analysis
            if self.sentence_model is None:
                logger.info("Loading sentence transformer model...")
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer loaded successfully")
                
            logger.info("All AI models initialized successfully")
            
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
                data = request.json
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
                
                # Start audio processing
                self.start_audio_processing()
                
                return jsonify({
                    "success": True,
                    "meeting_id": meeting_id,
                    "message": "Meeting started successfully"
                })
                
            except Exception as e:
                logger.error(f"Error starting meeting: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/stop_meeting', methods=['POST'])
        def stop_meeting():
            """Stop the current meeting session"""
            try:
                if self.current_meeting and self.current_meeting in self.meetings:
                    meeting = self.meetings[self.current_meeting]
                    meeting["end_time"] = datetime.now()
                    
                    # Generate final summary
                    summary = self.generate_meeting_summary(meeting)
                    meeting["summary"] = summary
                    
                    # Stop audio processing
                    self.stop_audio_processing()
                    
                    return jsonify({
                        "success": True,
                        "meeting_id": self.current_meeting,
                        "summary": summary
                    })
                else:
                    return jsonify({"success": False, "error": "No active meeting"})
                    
            except Exception as e:
                logger.error(f"Error stopping meeting: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/transcribe_audio', methods=['POST'])
        def transcribe_audio():
            """Transcribe audio chunk"""
            try:
                data = request.json
                audio_data = data.get('audio_data')  # Base64 encoded audio
                language = data.get('language', 'auto')
                
                if not audio_data:
                    return jsonify({"success": False, "error": "No audio data provided"})
                
                # Initialize models if needed
                if self.whisper_model is None:
                    self.initialize_models()
                
                # Decode and process audio
                audio_bytes = base64.b64decode(audio_data)
                transcription = self.process_audio_chunk(audio_bytes, language)
                
                if transcription:
                    # Store transcription
                    if self.current_meeting:
                        meeting = self.meetings[self.current_meeting]
                        meeting["transcriptions"].append({
                            "timestamp": datetime.now().isoformat(),
                            "text": transcription["text"],
                            "confidence": transcription.get("confidence", 0),
                            "language": transcription.get("language", "unknown")
                        })
                    
                    return jsonify({
                        "success": True,
                        "transcription": transcription
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "Could not transcribe audio"
                    })
                    
            except Exception as e:
                logger.error(f"Error transcribing audio: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/translate_text', methods=['POST'])
        def translate_text():
            """Translate text to target language"""
            try:
                data = request.json
                text = data.get('text')
                source_lang = data.get('source_lang', 'auto')
                target_lang = data.get('target_lang', 'en')
                
                if not text:
                    return jsonify({"success": False, "error": "No text provided"})
                
                translation = self.translate_text_to_language(text, source_lang, target_lang)
                
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
        
        @self.app.route('/detect_tasks', methods=['POST'])
        def detect_tasks():
            """Detect tasks and action items from text"""
            try:
                data = request.json
                text = data.get('text')
                
                if not text:
                    return jsonify({"success": False, "error": "No text provided"})
                
                # Initialize NLP model if needed
                if self.nlp_model is None:
                    self.initialize_models()
                
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
                data = request.json
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
        
        @self.app.route('/send_task_email', methods=['POST'])
        def send_task_email():
            """Send task assignment email"""
            try:
                data = request.json
                task = data.get('task')
                recipient_email = data.get('recipient_email')
                sender_email = data.get('sender_email')
                smtp_config = data.get('smtp_config', {})
                
                if not all([task, recipient_email, sender_email]):
                    return jsonify({"success": False, "error": "Missing required fields"})
                
                success = self.send_email_notification(task, recipient_email, sender_email, smtp_config)
                
                return jsonify({
                    "success": success,
                    "message": "Email sent successfully" if success else "Failed to send email"
                })
                
            except Exception as e:
                logger.error(f"Error sending email: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
    
    def process_audio_chunk(self, audio_bytes: bytes, language: str = "auto") -> Optional[Dict[str, Any]]:
        """Process a chunk of audio for transcription"""
        try:
            # Convert audio bytes to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Ensure audio is in the right format for Whisper (16kHz)
            if len(audio_data) == 0:
                return None
                
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Use Whisper for transcription
            if language == "auto":
                result = self.whisper_model.transcribe(audio_data)
            else:
                result = self.whisper_model.transcribe(audio_data, language=language)
            
            # Extract relevant information
            transcription = {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "confidence": self.calculate_confidence(result),
                "timestamp": datetime.now().isoformat()
            }
            
            return transcription if transcription["text"] else None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    def calculate_confidence(self, whisper_result: Dict) -> float:
        """Calculate confidence score from Whisper result"""
        try:
            # Whisper doesn't provide confidence directly, so we estimate it
            segments = whisper_result.get("segments", [])
            if not segments:
                return 0.5  # Default confidence
            
            # Average the no_speech_prob (inverse confidence)
            avg_no_speech = sum(seg.get("no_speech_prob", 0.5) for seg in segments) / len(segments)
            confidence = 1.0 - avg_no_speech
            
            return max(0.0, min(1.0, confidence))
        except Exception:
            return 0.5
    
    def translate_text_to_language(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using Transformers library"""
        try:
            if source_lang == target_lang or target_lang == "auto":
                return text
            
            # Create model key
            model_key = f"{source_lang}-{target_lang}"
            
            # Load translation model if not cached
            if model_key not in self.translation_models:
                try:
                    # Use Helsinki-NLP models for translation
                    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
                    
                    # Fallback to English if direct translation not available
                    if source_lang != "en" and target_lang != "en":
                        # Try via English
                        en_model_name = f"Helsinki-NLP/opus-mt-{source_lang}-en"
                        target_model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
                        
                        try:
                            # First translate to English
                            en_translator = pipeline("translation", model=en_model_name, device=0 if torch.cuda.is_available() else -1)
                            en_result = en_translator(text, max_length=512)[0]['translation_text']
                            
                            # Then translate to target
                            target_translator = pipeline("translation", model=target_model_name, device=0 if torch.cuda.is_available() else -1)
                            final_result = target_translator(en_result, max_length=512)[0]['translation_text']
                            
                            return final_result
                            
                        except Exception:
                            # If that fails, use a generic multilingual model
                            translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en", device=0 if torch.cuda.is_available() else -1)
                            result = translator(text, max_length=512)[0]['translation_text']
                            return result
                    else:
                        translator = pipeline("translation", model=model_name, device=0 if torch.cuda.is_available() else -1)
                        self.translation_models[model_key] = translator
                        
                except Exception as e:
                    logger.warning(f"Could not load translation model {model_name}: {e}")
                    # Fallback: just return original text with indicator
                    return f"[Translation to {target_lang}] {text}"
            
            # Perform translation
            translator = self.translation_models.get(model_key)
            if translator:
                result = translator(text, max_length=512)
                return result[0]['translation_text']
            else:
                return f"[Translation to {target_lang}] {text}"
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return f"[Translation Error] {text}"
    
    def extract_tasks_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tasks and action items from text using NLP"""
        tasks = []
        
        try:
            # Process text with spaCy
            doc = self.nlp_model(text)
            
            # Task detection patterns
            task_patterns = [
                # Direct action patterns
                r"(?:need to|have to|must|should|will|going to)\s+([^.!?]+)",
                # Assignment patterns
                r"(?:action item|task|todo|assignment):\s*([^.!?]+)",
                # Responsibility patterns
                r"(?:@\w+|assign|assigned to|responsible for)\s+([^.!?]+)",
                # Deadline patterns
                r"(?:by|due|deadline|before)\s+([^.!?]+)"
            ]
            
            import re
            
            # Extract potential tasks
            potential_tasks = []
            for pattern in task_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    task_text = match.group(1).strip()
                    if len(task_text) > 3:  # Filter out very short matches
                        potential_tasks.append({
                            "text": task_text,
                            "start": match.start(),
                            "end": match.end(),
                            "pattern": pattern
                        })
            
            # Process each potential task
            for i, potential_task in enumerate(potential_tasks):
                task_text = potential_task["text"]
                
                # Extract entities using spaCy
                task_doc = self.nlp_model(task_text)
                
                # Find people (assignees)
                assignee = None
                for ent in task_doc.ents:
                    if ent.label_ == "PERSON":
                        assignee = ent.text
                        break
                
                # Check for @mentions in surrounding context
                if not assignee:
                    context_start = max(0, potential_task["start"] - 100)
                    context_end = min(len(text), potential_task["end"] + 100)
                    context = text[context_start:context_end]
                    
                    mention_match = re.search(r"@(\w+)", context)
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
                    "source": "nlp_extraction",
                    "confidence": self.calculate_task_confidence(task_text)
                }
                
                tasks.append(task)
            
        except Exception as e:
            logger.error(f"Error extracting tasks: {e}")
        
        return tasks
    
    def determine_task_priority(self, text: str, position: int) -> str:
        """Determine task priority from context"""
        # Look at surrounding text (±200 characters)
        start = max(0, position - 200)
        end = min(len(text), position + 200)
        context = text[start:end].lower()
        
        # High priority indicators
        high_priority_words = self.config["priority_keywords"]["high"]
        if any(word in context for word in high_priority_words):
            return "high"
        
        # Low priority indicators
        low_priority_words = self.config["priority_keywords"]["low"]
        if any(word in context for word in low_priority_words):
            return "low"
        
        return "normal"
    
    def calculate_task_confidence(self, task_text: str) -> float:
        """Calculate confidence score for task extraction"""
        # Simple heuristic based on task text characteristics
        confidence = 0.5
        
        # Longer, more specific tasks get higher confidence
        if len(task_text.split()) >= 3:
            confidence += 0.2
        
        # Tasks with specific verbs get higher confidence
        action_verbs = ["complete", "finish", "create", "update", "review", "send", "call", "schedule"]
        if any(verb in task_text.lower() for verb in action_verbs):
            confidence += 0.2
        
        # Tasks with time indicators get higher confidence
        time_indicators = ["today", "tomorrow", "next week", "by", "before", "deadline"]
        if any(indicator in task_text.lower() for indicator in time_indicators):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def generate_meeting_summary(self, meeting: Dict[str, Any]) -> str:
        """Generate comprehensive meeting summary"""
        try:
            summary_parts = []
            
            # Header
            start_time = meeting.get("start_time", datetime.now())
            end_time = meeting.get("end_time", datetime.now())
            duration = end_time - start_time
            
            summary_parts.append("MEETING SUMMARY")
            summary_parts.append("=" * 50)
            summary_parts.append(f"Meeting ID: {meeting['id']}")
            summary_parts.append(f"Date: {start_time.strftime('%Y-%m-%d')}")
            summary_parts.append(f"Start Time: {start_time.strftime('%H:%M:%S')}")
            summary_parts.append(f"End Time: {end_time.strftime('%H:%M:%S')}")
            summary_parts.append(f"Duration: {str(duration).split('.')[0]}")
            summary_parts.append("")
            
            # Key discussion points
            transcriptions = meeting.get("transcriptions", [])
            if transcriptions:
                summary_parts.append("KEY DISCUSSION POINTS:")
                summary_parts.append("-" * 25)
                
                # Extract key sentences using sentence transformer
                key_points = self.extract_key_discussion_points(transcriptions)
                for i, point in enumerate(key_points, 1):
                    summary_parts.append(f"{i}. {point}")
                summary_parts.append("")
            
            # Action items
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
            
            # Participants (if available)
            participants = meeting.get("participants", [])
            if participants:
                summary_parts.append("PARTICIPANTS:")
                summary_parts.append("-" * 15)
                for participant in participants:
                    summary_parts.append(f"• {participant}")
                summary_parts.append("")
            
            # Full transcription
            if transcriptions:
                summary_parts.append("FULL TRANSCRIPTION:")
                summary_parts.append("-" * 20)
                
                for transcript in transcriptions:
                    timestamp = datetime.fromisoformat(transcript["timestamp"]).strftime("%H:%M:%S")
                    summary_parts.append(f"[{timestamp}] {transcript['text']}")
                
            return "\\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating meeting summary."
    
    def extract_key_discussion_points(self, transcriptions: List[Dict]) -> List[str]:
        """Extract key discussion points using sentence similarity"""
        try:
            # Combine all transcriptions
            all_text = " ".join([t["text"] for t in transcriptions])
            
            # Split into sentences
            sentences = []
            for sentence in all_text.split('.'):
                sentence = sentence.strip()
                if len(sentence) > 20:  # Filter out very short sentences
                    sentences.append(sentence)
            
            if not sentences:
                return []
            
            # Use sentence transformer to find most important sentences
            if self.sentence_model and len(sentences) > 3:
                embeddings = self.sentence_model.encode(sentences)
                
                # Calculate centrality scores (sentences similar to the average)
                mean_embedding = np.mean(embeddings, axis=0)
                similarities = [np.dot(emb, mean_embedding) / (np.linalg.norm(emb) * np.linalg.norm(mean_embedding)) 
                              for emb in embeddings]
                
                # Get top sentences
                top_indices = np.argsort(similarities)[-5:][::-1]  # Top 5 most central sentences
                key_points = [sentences[i] for i in top_indices]
            else:
                # Fallback: use keyword-based selection
                key_points = self.extract_key_points_by_keywords(sentences)
            
            return key_points[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return []
    
    def extract_key_points_by_keywords(self, sentences: List[str]) -> List[str]:
        """Fallback method to extract key points using keywords"""
        important_keywords = [
            "decision", "decided", "conclude", "important", "key", "main",
            "problem", "issue", "solution", "agree", "disagree", "vote",
            "budget", "cost", "deadline", "milestone", "goal", "objective",
            "next steps", "follow up", "action required"
        ]
        
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for keyword in important_keywords if keyword in sentence.lower())
            if score > 0:
                scored_sentences.append((score, sentence))
        
        # Sort by score and return top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return [sentence for score, sentence in scored_sentences[:5]]
    
    def send_email_notification(self, task: Dict[str, Any], recipient_email: str, 
                              sender_email: str, smtp_config: Dict[str, Any]) -> bool:
        """Send email notification for task assignment"""
        try:
            # Email content
            subject = f"Task Assignment: {task['description'][:50]}..."
            
            body = f"""
Dear {task.get('assignee', 'Team Member')},

You have been assigned a new task from the meeting:

Task: {task['description']}
Priority: {task.get('priority', 'Normal')}
Assigned Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Task ID: {task.get('id', 'N/A')}

Please acknowledge receipt of this task and provide updates as needed.

Best regards,
Meeting Assistant System
            """
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # SMTP configuration
            smtp_server = smtp_config.get('server', 'localhost')
            smtp_port = smtp_config.get('port', 587)
            smtp_username = smtp_config.get('username', sender_email)
            smtp_password = smtp_config.get('password', '')
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            
            if smtp_password:
                server.login(smtp_username, smtp_password)
            
            text = msg.as_string()
            server.sendmail(sender_email, recipient_email, text)
            server.quit()
            
            logger.info(f"Email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def start_audio_processing(self):
        """Start background audio processing thread"""
        self.is_processing = True
        self.audio_processor_thread = threading.Thread(target=self._audio_processing_loop)
        self.audio_processor_thread.daemon = True
        self.audio_processor_thread.start()
        logger.info("Audio processing thread started")
    
    def stop_audio_processing(self):
        """Stop background audio processing"""
        self.is_processing = False
        if self.audio_processor_thread:
            self.audio_processor_thread.join(timeout=5)
        logger.info("Audio processing thread stopped")
    
    def _audio_processing_loop(self):
        """Background thread for processing audio queue"""
        while self.is_processing:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get(timeout=1)
                    # Process audio data here
                    # This would be called by the Chrome extension
                else:
                    time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
    
    def run(self, host='localhost', port=5000, debug=False):
        """Run the Flask server"""
        logger.info(f"Starting Meeting Transcription Backend on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def main():
    """Main entry point"""
    backend = MeetingTranscriptionBackend()
    
    # Pre-load models on startup for faster first use
    logger.info("Pre-loading AI models on startup...")
    try:
        backend.initialize_models()
        logger.info("✓ All AI models loaded successfully on startup")
    except Exception as e:
        logger.warning(f"Could not pre-load models on startup (they will load on first use): {e}")
    
    # Run server
    try:
        backend.run(host='localhost', port=5000, debug=False)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()