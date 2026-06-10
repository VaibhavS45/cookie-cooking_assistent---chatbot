"""
Voice Chef module - Speech-to-Text and Text-to-Speech for Streamlit.
Adds voice interaction capabilities for impressive demo presentations.
"""

import io
import os
from typing import Optional
import wave
import tempfile

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class VoiceChef:
    """Voice interaction for Chef Annapurna AI."""
    
    @staticmethod
    def is_speech_available() -> bool:
        return SPEECH_AVAILABLE
    
    @staticmethod
    def is_tts_available() -> bool:
        return TTS_AVAILABLE
    
    @staticmethod
    def speech_to_text(audio_bytes: Optional[bytes] = None, 
                      audio_file: Optional[str] = None,
                      timeout: int = 5) -> str:
        """
        Convert speech to text using microphone or audio file.
        
        Args:
            audio_bytes: Raw audio bytes from Streamlit recorder
            audio_file: Path to audio file
            timeout: Max listening time in seconds
            
        Returns:
            Transcribed text
        """
        if not SPEECH_AVAILABLE:
            return "Voice recognition module not installed. Run: pip install SpeechRecognition"
        
        recognizer = sr.Recognizer()
        
        try:
            if audio_bytes:
                # Process raw audio bytes
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                
                with sr.AudioFile(tmp_path) as source:
                    audio = recognizer.record(source)
                os.unlink(tmp_path)
            
            elif audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)
            
            else:
                # Use microphone
                with sr.Microphone() as source:
                    print("🎤 Listening... (speak now)")
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=timeout)
            
            # Use Google's free speech recognition
            text = recognizer.recognize_google(audio)
            return text
            
        except sr.WaitTimeoutError:
            return "⏱️ No speech detected within timeout period."
        except sr.UnknownValueError:
            return "🤔 Could not understand audio. Please speak clearly."
        except sr.RequestError as e:
            return f"🌐 Speech recognition service error: {e}"
        except Exception as e:
            return f"⚠️ Error: {str(e)}"
    
    @staticmethod
    def text_to_speech(text: str, rate: int = 180, slow: bool = False) -> Optional[bytes]:
        """
        Convert text to speech audio bytes.
        
        Args:
            text: Text to speak
            rate: Speaking rate (words per minute)
            slow: Whether to speak slowly
            
        Returns:
            Audio bytes in WAV format, or None if TTS unavailable
        """
        if not TTS_AVAILABLE:
            return None
        
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', rate if not slow else rate // 2)
            
            # Get available voices and pick a feminine one for Chef Annapurna
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            
            # Save to temporary file and read back
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
            
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            
            with open(tmp_path, 'rb') as f:
                audio_bytes = f.read()
            
            os.unlink(tmp_path)
            return audio_bytes
            
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
    
    @staticmethod
    def create_audio_player(audio_bytes: bytes) -> str:
        """
        Create an HTML audio player for Streamlit.
        
        Args:
            audio_bytes: WAV audio bytes
            
        Returns:
            HTML string for audio playback
        """
        import base64
        audio_base64 = base64.b64encode(audio_bytes).decode()
        return f"""
            <audio controls autoplay style="width: 100%;">
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                Your browser does not support audio playback.
            </audio>
        """


if __name__ == "__main__":
    print(f"● Speech Recognition: {'✅ Available' if SPEECH_AVAILABLE else '❌ Not installed'}")
    print(f"● Text-to-Speech:     {'✅ Available' if TTS_AVAILABLE else '❌ Not installed'}")
    
    # Test TTS
    if TTS_AVAILABLE:
        print("\n🔊 Testing chef voice...")
        audio = VoiceChef.text_to_speech("Namaste! Chef Annapurna here. Let's cook something delicious!", slow=True)
        if audio:
            print(f"   Audio generated: {len(audio)} bytes")
    
    if SPEECH_AVAILABLE:
        print("\n🎤 Listening test (speak now)...")
        text = VoiceChef.speech_to_text(timeout=3)
        print(f"   You said: {text}")