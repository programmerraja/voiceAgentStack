import base64
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from pipecat.frames.frames import (
    Frame,
    DataFrame,
    SystemFrame,
    InputAudioRawFrame,
    LLMTextFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TransportMessageUrgentFrame,
    # Add more as needed
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# --- New Frame for Conversation Initiation ---
@dataclass
class ElevenLabsConversationInitiationFrame(SystemFrame):
    """Frame for ElevenLabs conversation_initiation_client_data event."""
    conversation_config_override: Dict[str, Any]
    custom_llm_extra_body: Optional[Dict[str, Any]] = None
    dynamic_variables: Optional[Dict[str, Any]] = None

# --- Observer Params ---
@dataclass
class ElevenLabsObserverParams:
    bot_llm_enabled: bool = True
    bot_tts_enabled: bool = True
    bot_speaking_enabled: bool = True
    user_llm_enabled: bool = True
    user_speaking_enabled: bool = True
    user_transcription_enabled: bool = True
    # Add more toggles as needed

# --- Processor ---
class ElevenLabsProcessor(FrameProcessor):
    """
    Handles conversion between ElevenLabs protocol messages and Pipecat frames.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add any state needed

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # OUT: Convert Pipecat frames to ElevenLabs messages
        # IN: Convert ElevenLabs messages to Pipecat frames
        # TODO: Implement routing logic
        pass

    async def handle_incoming_message(self, message: Dict[str, Any]):
        """
        Convert incoming ElevenLabs message to Pipecat frame and push to pipeline.
        """
        msg_type = message.get("type")
        if msg_type == "conversation_initiation_client_data":
            frame = ElevenLabsConversationInitiationFrame(
                conversation_config_override=message.get("conversation_config_override", {}),
                custom_llm_extra_body=message.get("custom_llm_extra_body"),
                dynamic_variables=message.get("dynamic_variables"),
            )
            await self.push_frame(frame)
        elif msg_type == "user_audio_chunk":
            audio_b64 = message.get("user_audio_chunk")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                # TODO: Use correct sample_rate/num_channels from context or config
                frame = InputAudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1)
                await self.push_frame(frame)
        elif msg_type == "user_message":
            text = message.get("text", "")
            frame = LLMTextFrame(text=text)
            await self.push_frame(frame)
        elif msg_type == "contextual_update":
            # TODO: Create and push a new frame for contextual_update
            pass
        elif msg_type == "pong":
            # TODO: Handle pong (optional)
            pass
        elif msg_type == "user_activity":
            # TODO: Handle user_activity (optional)
            pass
        # TODO: Handle other message types as needed
        else:
            # Unknown or unhandled message type
            pass

    # Helper: Convert Pipecat frame to ElevenLabs message
    def frame_to_elevenlabs_message(self, frame: Frame) -> Optional[Dict[str, Any]]:
        # OUT: Map Pipecat frames to ElevenLabs protocol messages
        if isinstance(frame, TranscriptionFrame):
            return {
                "type": "user_transcript",
                "user_transcription_event": {
                    "user_transcript": frame.text
                }
            }
        elif isinstance(frame, InterimTranscriptionFrame):
            # ElevenLabs does not have a direct interim transcript, may skip or map as needed
            pass
        elif isinstance(frame, LLMTextFrame):
            return {
                "type": "agent_response",
                "agent_response_event": {
                    "agent_response": frame.text
                }
            }
        elif isinstance(frame, OutputAudioRawFrame):
            return {
                "type": "audio",
                "audio_event": {
                    "audio_base_64": base64.b64encode(frame.audio).decode(),
                    "event_id": 1  # TODO: Generate/track event_id
                }
            }
        elif isinstance(frame, TTSStartedFrame):
            # ElevenLabs does not have a direct TTS started event, skip
            pass
        elif isinstance(frame, TTSStoppedFrame):
            # ElevenLabs does not have a direct TTS stopped event, skip
            pass
        elif isinstance(frame, BotStartedSpeakingFrame):
            # ElevenLabs does not have a direct bot started speaking event, skip
            pass
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # ElevenLabs does not have a direct bot stopped speaking event, skip
            pass
        # TODO: Map other frame types (e.g., interruption, vad_score, etc.)
        return None

# --- Observer ---
class ElevenLabsObserver(BaseObserver):
    """
    Pipeline frame observer for ElevenLabs protocol message handling.
    Converts pipeline frames into ElevenLabs-compatible messages for the client.
    """
    def __init__(self, processor: FrameProcessor, *, params: Optional[ElevenLabsObserverParams] = None, **kwargs):
        super().__init__(**kwargs)
        self._processor = processor
        self._params = params or ElevenLabsObserverParams()
        self._frames_seen = set()
        self._bot_transcription = ""

    async def on_push_frame(self, data: FramePushed):
        src = data.source
        frame = data.frame
        direction = data.direction

        if frame.id in self._frames_seen:
            return
        mark_as_seen = True

        if (
            isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame))
            and self._params.user_speaking_enabled
        ):
            await self._handle_user_interruptions(frame)
        elif (
            isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame))
            and direction == FrameDirection.UPSTREAM
            and self._params.bot_speaking_enabled
        ):
            await self._handle_bot_speaking(frame)
        elif (
            isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame))
            and self._params.user_transcription_enabled
        ):
            await self._handle_user_transcriptions(frame)
        elif isinstance(frame, LLMTextFrame) and self._params.bot_llm_enabled:
            await self._handle_llm_text_frame(frame)
        elif isinstance(frame, OutputAudioRawFrame) and self._params.bot_tts_enabled:
            await self._handle_audio(frame)
        # TODO: Add more frame handlers as needed (e.g., for contextual_update, vad_score, etc.)

        if mark_as_seen:
            self._frames_seen.add(frame.id)

    async def push_transport_message_urgent(self, message: Dict[str, Any]):
        frame = TransportMessageUrgentFrame(message=message)
        await self._processor.push_frame(frame)

    async def _handle_user_interruptions(self, frame: Frame):
        # ElevenLabs: interruption event (user started/stopped speaking)
        if isinstance(frame, UserStartedSpeakingFrame):
            message = {"type": "vad_score", "vad_score_event": {"vad_score": 1.0}}  # Example
            await self.push_transport_message_urgent(message)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            message = {"type": "vad_score", "vad_score_event": {"vad_score": 0.0}}  # Example
            await self.push_transport_message_urgent(message)
        # TODO: Map to interruption event if needed

    async def _handle_bot_speaking(self, frame: Frame):
        # ElevenLabs does not have direct bot speaking events, but could map to interruption if needed
        pass

    async def _handle_llm_text_frame(self, frame: LLMTextFrame):
        # Final agent response
        message = {
            "type": "agent_response",
            "agent_response_event": {"agent_response": frame.text}
        }
        await self.push_transport_message_urgent(message)
        self._bot_transcription += frame.text
        # Optionally, send internal_tentative_agent_response for partials
        # TODO: If you have partials, emit "internal_tentative_agent_response"

    async def _handle_user_transcriptions(self, frame: Frame):
        if isinstance(frame, TranscriptionFrame):
            message = {
                "type": "user_transcript",
                "user_transcription_event": {"user_transcript": frame.text}
            }
            await self.push_transport_message_urgent(message)
        elif isinstance(frame, InterimTranscriptionFrame):
            message = {
                "type": "internal_tentative_agent_response",
                "tentative_agent_response_internal_event": {"tentative_agent_response": frame.text}
            }
            await self.push_transport_message_urgent(message)

    async def _handle_audio(self, frame: OutputAudioRawFrame):
        # Agent's audio stream
        message = {
            "type": "audio",
            "audio_event": {
                "audio_base_64": base64.b64encode(frame.audio).decode(),
                "event_id": 1  # TODO: Track/generate event_id
            }
        }
        await self.push_transport_message_urgent(message)

# --- Helpers for conversion (if needed externally) ---
def elevenlabs_message_to_frame(message: Dict[str, Any]) -> Optional[Frame]:
    # Standalone helper for converting incoming ElevenLabs messages to frames
    # (for use in tests or other entry points)
    # TODO: Implement as in handle_incoming_message
    pass

def frame_to_elevenlabs_message(frame: Frame) -> Optional[Dict[str, Any]]:
    # Standalone helper for converting frames to ElevenLabs messages
    # (for use in tests or other entry points)
    # TODO: Implement as in frame_to_elevenlabs_message method
    pass
