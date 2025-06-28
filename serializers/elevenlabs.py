import audioop
import base64
import json
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from frameworks.elevenlabs import ElevenLabsConversationInitiationFrame
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class ElevenLabsFrameSerializer(FrameSerializer):
    """Serializer for ElevenLabs Conversational AI WebSocket protocol.
    Handles conversion between Pipecat frames and ElevenLabs protocol messages.
    Supports audio resampling if needed.
    """

    class InputParams(BaseModel):
        """Configuration parameters for ElevenLabsFrameSerializer.

        Attributes:
            sample_rate: Sample rate used by ElevenLabs, defaults to 16000 Hz.
            num_channels: Number of channels used by ElevenLabs, defaults to 1.
        """

        sample_rate: int = 24000
        num_channels: int = 1
        audio_format: Literal["pcm", "ulaw"] = "pcm"

    def __init__(self, params: Optional[InputParams] = None):
        self._params = params or ElevenLabsFrameSerializer.InputParams()
        self._sample_rate = self._params.sample_rate
        self._num_channels = self._params.num_channels
        self._audio_format = self._params.audio_format
        self._resampler = create_default_resampler()

    @property
    def type(self) -> FrameSerializerType:
        return FrameSerializerType.TEXT

    async def setup(self, frame: StartFrame):
        self._sample_rate = frame.audio_in_sample_rate or self._params.sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Convert Pipecat frame to ElevenLabs WebSocket message (JSON string).
        This is for server->client direction.
        """
        
        if isinstance(frame, OutputAudioRawFrame):
            # Outbound agent audio (server->client)
            audio = frame.audio
            # if frame.sample_rate != self._params.sample_rate or frame.num_channels != self._params.num_channels:
            #     audio = await self._resampler.resample(audio, frame.sample_rate, self._params.sample_rate)
            if self._audio_format == "ulaw":
                audio = await pcm_to_ulaw(audio, frame.sample_rate, self._params.sample_rate, self._resampler)
            message = {
                "type": "audio",
                "audio_event": {
                    "audio_base_64": base64.b64encode(audio).decode("utf-8"),
                    "event_id": 1  # TODO: Track/generate event_id
                }
            }
            return json.dumps(message)
        elif isinstance(frame, TranscriptionFrame):
            message = {
                "type": "user_transcript",
                "user_transcription_event": {
                    "user_transcript": frame.text
                }
            }
            return json.dumps(message)
        elif isinstance(frame, InterimTranscriptionFrame):
            message = {
                "type": "internal_tentative_agent_response",
                "tentative_agent_response_internal_event": {
                    "tentative_agent_response": frame.text
                }
            }
            return json.dumps(message)
        
        elif isinstance(frame, LLMTextFrame):
            # By default, treat as agent response (server->client)
            # If you want to distinguish user vs agent, you may need to add a custom field or context
            message = {
                "type": "agent_response",
                "agent_response_event": {
                    "agent_response": frame.text
                }
            }
            return json.dumps(message)
        
        elif isinstance(frame, ElevenLabsConversationInitiationFrame):
            message = {
                "type": "conversation_initiation_client_data",
                "conversation_config_override": frame.conversation_config_override,
            }
            if frame.custom_llm_extra_body:
                message["custom_llm_extra_body"] = frame.custom_llm_extra_body
            if frame.dynamic_variables:
                message["dynamic_variables"] = frame.dynamic_variables
            return json.dumps(message)
       
        elif isinstance(frame, TransportMessageFrame) or isinstance(frame, TransportMessageUrgentFrame):
            # Pass through if already a message dict
            return json.dumps(frame.message)
        # TODO: Add more frame types as needed (vad_score, interruption, etc)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """
        Convert ElevenLabs WebSocket message (JSON string) to Pipecat frame.
        This is for client->server direction.
        """
        if isinstance(data, bytes):
            data = data.decode()
        try:
            message = json.loads(data)
        except Exception as e:
            logger.warning(f"Failed to parse ElevenLabs message: {e}")
            return None

        msg_type = message.get("type")
        # Handle user_audio_chunk (client->server)
        if "user_audio_chunk" in message:
            audio_b64 = message["user_audio_chunk"]
            audio_bytes = base64.b64decode(audio_b64)
            if self._audio_format == "ulaw":
                audio_bytes = await ulaw_to_pcm(audio_bytes, 8000, self._sample_rate, self._resampler)
                
            return InputAudioRawFrame(audio=audio_bytes, sample_rate=self._sample_rate, num_channels=self._num_channels)
        #Handle it later
        # if msg_type == "user_message":
        #     text = message.get("text", "")
        #     # By default, treat as LLMTextFrame (user input)
        #     return LLMTextFrame(text=text)
        
        # if msg_type == "audio":
        #     audio_b64 = message.get("audio_event", {}).get("audio_base_64")
        #     if audio_b64:
        #         audio_bytes = base64.b64decode(audio_b64)
        #         return InputAudioRawFrame(audio=audio_bytes, sample_rate=self._sample_rate, num_channels=self._num_channels)
        # elif msg_type == "user_transcript":
        #     text = message.get("user_transcription_event", {}).get("user_transcript", "")
        #     return TranscriptionFrame(text=text, user_id="", timestamp="")
        # elif msg_type == "internal_tentative_agent_response":
        #     text = message.get("tentative_agent_response_internal_event", {}).get("tentative_agent_response", "")
        #     return InterimTranscriptionFrame(text=text, user_id="", timestamp="")
        # elif msg_type == "agent_response":
        #     text = message.get("agent_response_event", {}).get("agent_response", "")
        #     return LLMTextFrame(text=text)
        
        
        elif msg_type == "conversation_initiation_client_data":
            return ElevenLabsConversationInitiationFrame(
                conversation_config_override=message.get("conversation_config_override", {}),
                custom_llm_extra_body=message.get("custom_llm_extra_body"),
                dynamic_variables=message.get("dynamic_variables"),
            )
        return None
