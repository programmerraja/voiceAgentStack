
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    SystemFrame,
    TransportMessageUrgentFrame,
    LLMMessagesUpdateFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pydantic import BaseModel

from pydantic import BaseModel
from typing import List, Dict, Optional



class BuiltInTool(BaseModel):
    type: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None


class KnowledgeBaseItem(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None
    usage_mode: Optional[str] = None


class RagConfig(BaseModel):
    enabled: Optional[bool] = None
    embedding_model: Optional[str] = None
    max_retrieved_rag_chunks_count: Optional[int] = None
    max_documents_length: Optional[int] = None
    max_vector_distance: float


class Prompt(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    llm: Optional[str] = None
    tool_ids: Optional[List[str]] = None
    built_in_tools: Optional[Dict[str, Optional[BuiltInTool]]] = None
    knowledge_base: Optional[List[KnowledgeBaseItem]] = None
    mcp_server_ids: Optional[List[str]] = None
    native_mcp_server_ids: Optional[List[str]] = None
    custom_llm: Optional[str] = None
    rag: Optional[RagConfig] = None


class Agent(BaseModel):
    prompt: Prompt
    first_message: Optional[str] = None
    language: Optional[str] = None
    # dynamic_variables: Optional[Dict[str, Any]] = None


class ASR(BaseModel):
    quality: Optional[str] = None
    provider: Optional[str] = None
    user_input_audio_format: Optional[str] = None
    keywords: Optional[List[str]] = None


class TTS(BaseModel):
    voice_id: Optional[str] = None
    supported_voices: Optional[List[str]] = None
    model_id: Optional[str] = None
    agent_output_audio_format: Optional[str] = None
    optimize_streaming_latency: Optional[int] = None
    stability: Optional[float] = None
    speed: Optional[float] = None
    similarity_boost: Optional[float] = None
    pronunciation_dictionary_locators: Optional[List[str]] = None


class Turn(BaseModel):
    turn_timeout: Optional[int] = None
    silence_end_call_timeout: Optional[int] = None


class Conversation(BaseModel):
    max_duration_seconds: Optional[int] = None
    text_only: Optional[bool] = None
    client_events: Optional[List[str]] = None


class ConversationConfig(BaseModel):
    agent: Optional[Agent] = None
    asr: Optional[ASR] = None
    tts: Optional[TTS] = None
    turn: Optional[Turn] = None
    conversation: Optional[Conversation] = None
    language_presets: Optional[Dict[str, str]] = None
  


class Evaluation(BaseModel):
    criteria: List[str]


class Auth(BaseModel):
    enable_auth: bool
    allowlist: List[str]


class ConversationConfigOverride(BaseModel):
    agent: Optional[Dict[str, Optional[bool]]] = None
    tts: Optional[Dict[str, Optional[bool]]] = None
    conversation: Optional[Dict[str, Optional[bool]]] = None


class Overrides(BaseModel):
    enable_conversation_initiation_client_data_from_webhook: Optional[bool] = None
    custom_llm_extra_body: Optional[bool] = None
    conversation_config_override: Optional[ConversationConfigOverride] = None


class CallLimits(BaseModel):
    agent_concurrency_limit: Optional[int] = None
    bursting_enabled: Optional[bool] = None
    daily_limit: Optional[int] = None


class Privacy(BaseModel):
    record_voice: Optional[bool] = None
    retention_days: Optional[int] = None
    delete_transcript_and_pii: Optional[bool] = None
    delete_audio: Optional[bool] = None
    apply_to_existing_conversations: Optional[bool] = None
    zero_retention_mode: Optional[bool] = None


class PlatformSettings(BaseModel):
    overrides: Optional[Overrides] = None
    call_limits: Optional[CallLimits] = None
    privacy: Optional[Privacy] = None
    data_collection: Optional[Dict[str, str]] = None
    workspace_overrides: Optional[Dict[str, str]] = None


class RootModel(BaseModel):
    name: Optional[str] = None
    conversation_config: Optional[ConversationConfig] = None
    platform_settings: Optional[PlatformSettings] = None


@dataclass
class ElevenLabsConversationInitiationFrame(SystemFrame):
    """Frame for ElevenLabs conversation_initiation_client_data event."""
    conversation_config_override: Optional[ConversationConfig] = None   
    custom_llm_extra_body: Optional[Dict[str, Any]] = None
    dynamic_variables: Optional[Dict[str, Any]] = None


@dataclass
class ElevenLabsProcessorParams(ConversationConfig):
    pass


class ElevenLabsProcessor(FrameProcessor):
    """
    Handles conversion between ElevenLabs protocol messages and Pipecat frames.
    """

    def __init__(self, params: Optional[ConversationConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.conversation_config = params if params else ConversationConfig()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames through the RTVI processor.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, ElevenLabsConversationInitiationFrame):
            await self._handle_conversation_initiation(frame)
        else:
            await self.push_frame(frame, direction)

    async def _handle_conversation_initiation(
        self, frame: ElevenLabsConversationInitiationFrame
    ):
        """Handle conversation initiation frame."""
        conversation_config = frame.conversation_config_override or self.conversation_config
        agent_config = conversation_config.agent 
       
        if agent_config:
            agent_prompt = (
                agent_config.prompt.prompt
                or self.conversation_config.agent.prompt.prompt
            )
            # agent_first_message = (
            #     agent_config.first_message
            #     or self.conversation_config.agent.first_message
            # )
        else:
            agent_prompt = self.conversation_config.agent.prompt.prompt
            # agent_first_message = self.conversation_config.agent.first_message
            # custom_llm_extra_body = frame.custom_llm_extra_body 

        dynamic_variables = frame.dynamic_variables 
        
        def render_double_brace(template: str, variables: dict) -> str:
            pattern = re.compile(r"\{\{(\w+)\}\}")
            def replacer(match):
                var_name = match.group(1)
                return str(variables.get(var_name, match.group(0)))

            return pattern.sub(replacer, template)
        
        try:
            filled_prompt = render_double_brace(agent_prompt, dynamic_variables)
        except Exception:
            filled_prompt = agent_prompt
        llm_update_frame = LLMMessagesUpdateFrame(messages=[{"role": "system", "content": filled_prompt}])
        await self.push_frame(llm_update_frame, FrameDirection.UPSTREAM)
        # await self.push_frame(llm_update_frame, FrameDirection.DOWNSTREAM)

        # # Continue with the rest of the initiation logic
        # conversation_initiation_frame = ElevenLabsConversationInitiationFrame(
        #     conversation_config_override=context,
        #     custom_llm_extra_body=custom_llm_extra_body,
        #     dynamic_variables=dynamic_variables,
        # )
        # await self.push_frame(conversation_initiation_frame)
        
        



