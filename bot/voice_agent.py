import os
import json
from loguru import logger
import time
import types

# Monkeypatch timing for pipecat core classes
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

# # Utility to time all methods of a class
# def time_all_methods(cls):
#     for attr_name in dir(cls):
#         if attr_name.startswith("__"):
#             continue  # skip dunder methods
#         attr = getattr(cls, attr_name)
#         if isinstance(attr, types.FunctionType):
#             def make_timed_method(method, name):
#                 def timed(self, *args, **kwargs):
#                     start = time.perf_counter()
#                     result = method(self, *args, **kwargs)
#                     duration = time.perf_counter() - start
#                     print(f"[TIMING] {cls.__name__}.{name} took {duration:.2f} seconds")
#                     return result
#                 return timed
#             setattr(cls, attr_name, make_timed_method(attr, attr_name))
#         elif isinstance(attr, types.MethodType) and hasattr(attr, "__func__") and hasattr(attr.__func__, "__code__") and attr.__func__.__code__.co_flags & 0x80:
#             # async def (coroutine)
#             def make_timed_async_method(method, name):
#                 async def timed(self, *args, **kwargs):
#                     start = time.perf_counter()
#                     result = await method(self, *args, **kwargs)
#                     duration = time.perf_counter() - start
#                     print(f"[TIMING] {cls.__name__}.{name} took {duration:.2f} seconds")
#                     return result
#                 return timed
#             setattr(cls, attr_name, make_timed_async_method(attr, attr_name))

# Apply timing to all methods of these classes
# time_all_methods(Pipeline)
# time_all_methods(PipelineTask)
# time_all_methods(PipelineRunner)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from vad.webrtcvads.webrtcvads import WebRTCVADAnalyzer

# from vad.tenvad.tenvad import TENVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from pipecat.services.openai.llm import OpenAILLMService
import cProfile

from frameworks.elevenlabs import (
    ElevenLabsProcessor,
    ConversationConfig,
)
from serializers.elevenlabs import ElevenLabsFrameSerializer
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.services.whisper.stt import WhisperSTTService
from service.whisper.stt import WebSocketSTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.openai.stt import OpenAISTTService

# Alternative streaming STT service for even lower latency
# from service.parakeet.stt import ParakeetStreamingSTTService
from service.chatterbot.tts import ChatterboxTTSService

# Add imports for frame types used in event handlers
from pipecat.frames.frames import (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)

from pipecat.utils.tracing.setup import setup_tracing
from tools.tools import build_tools, json_to_tools_schema, WebhookTool
from pipecat.adapters.schemas.function_schema import FunctionSchema
from typing import Dict, Optional, Any
from pipecat.services.llm_service import FunctionCallRegistryItem, FunctionCallParams
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.groq.llm import GroqLLMService

from pipecat.services.azure.tts import AzureTTSService
from pipecat.services.azure.stt import AzureSTTService
# time_all_methods(FastAPIWebsocketTransport)

# Optimize VAD analyzer for faster response
vad_analyzer = SileroVADAnalyzer()

IS_TRACING_ENABLED = bool(os.getenv("ENABLE_TRACING"))

# Initialize tracing if enabled
if IS_TRACING_ENABLED:
    # Create the exporter
    otlp_exporter = OTLPSpanExporter()

    # Set up tracing with the exporter
    setup_tracing(
        service_name="voice-agent-stack",
        exporter=otlp_exporter,
        console_export=bool(os.getenv("OTEL_CONSOLE_EXPORT")),
    )
    logger.info("OpenTelemetry tracing initialized")


class VoiceAgent:
    def __init__(
        self,
        websocket_client,
        conversation_id="voice-agent-conversation-1",
    ):
        init_start = time.perf_counter()
        self.websocket_client = websocket_client

        self.tools_config_path = os.path.join(
            os.path.dirname(__file__), "../config/tools.json"
        )
        self.conversation_config_path = os.path.join(
            os.path.dirname(__file__), "../config/conversations.json"
        )
        self.context_messages = []
        self.tracing_enabled = IS_TRACING_ENABLED
        self.conversation_id = conversation_id
        self.pipeline = None
        self.task = None
        self.tools = []
        self.conversation_config = None
        self.runner = None
        self.ws_transport = None
        self.context_aggregator = None
        self._functions: Dict[Optional[str], FunctionCallRegistryItem] = {}
        # self._load_tools()
        logger.info("Loading conversation config...")
        config_start = time.perf_counter()
        self._load_conversation_config()
        logger.info(
            f"Loaded conversation config in {time.perf_counter() - config_start:.2f} seconds"
        )
        logger.info(
            f"VoiceAgent.__init__ completed in {time.perf_counter() - init_start:.2f} seconds"
        )

    def _load_tools(self):
        tools_start = time.perf_counter()
        with open(self.tools_config_path) as f:
            self.tools_json = json.load(f)
        self.tools = build_tools(self.tools_json)
        logger.info(f"Loaded tools in {time.perf_counter() - tools_start:.2f} seconds")

    def _load_conversation_config(self):
        with open(self.conversation_config_path) as f:
            config = json.load(f)
            self.conversation_config = ConversationConfig(
                **config["conversation_config"]
            )
            self.context_messages.append(
                {
                    "role": "system",
                    "content": self.conversation_config.agent.prompt.prompt,
                }
            )

    def _build_context_aggregator(self, tools_schema):
        context = OpenAILLMContext(
            self.context_messages, tools=ToolsSchema(standard_tools=tools_schema)
        )
        return self.llm.create_context_aggregator(context)

    def _setup_pipeline(self):
        setup_start = time.perf_counter()
        logger.info("Setting up FastAPIWebsocketTransport...")
        ws_start = time.perf_counter()
        self.ws_transport = FastAPIWebsocketTransport(
            websocket=self.websocket_client,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_analyzer=vad_analyzer,
                # vad_analyzer=WebRTCVADAnalyzer(),
                # vad_analyzer=TENVADAnalyzer(),
                serializer=ElevenLabsFrameSerializer(
                    params=ElevenLabsFrameSerializer.InputParams(
                        audio_format="ulaw", sample_rate=8000
                    )
                ),
            ),
        )
        logger.info(
            f"FastAPIWebsocketTransport setup in {time.perf_counter() - ws_start:.2f} seconds"
        )

        logger.info("Initializing WhisperSTTService...")

        stt_start = time.perf_counter()

        # Use local Whisper for lowest latency (no network calls)
        self.stt = WebSocketSTTService(
            ws_url="ws://localhost:9800",
            language="en",
        )

        # self.stt = WhisperLiveSTTService(
        #     sample_rate=24000,
        # )

        # Alternative: Use ParakeetStreamingSTTService for even lower latency
        # self.stt = ParakeetStreamingSTTService(
        #     model_name="nvidia/parakeet-tdt-0.6b-v2",
        #     sample_rate=24000,
        #     chunk_duration_ms=50,  # 50ms chunks for low latency
        #     enable_partial_results=True,
        # )

        # Alternative: Use OpenAI STT (has network delay)
        # self.stt = OpenAISTTService(
        #     base_url="http://localhost:8000/v1",
        #     api_key="not-needed",
        #     model='Systran/faster-whisper-small',
        #     language="en",
        #     sample_rate=24000,
        # )

        # Alternative: Use Azure STT with debug logging
        # self.stt = AzureSTTService (
        #     api_key=os.getenv("AZURE_SPEACH_API_KEY"),
        #     endpoint=os.getenv("AZURE_SPEACH_ENDPOINT"),
        #     region="eastus",
        #     sample_rate=24000,
        # )

        logger.info(
            f"WhisperSTTService initialized in {time.perf_counter() - stt_start:.2f} seconds"
        )

        # self.stt  = OpenAISTTService(
        #     base_url="http://localhost:8005/v1",
        #     api_key="not-needed",
        #     # model="Systran/faster-whisper-large-v3",
        #     model="Systran/faster-distil-whisper-large-v3",
        #     language="en",
        #     sample_rate=24000,
        # )
        # self.llm = OpenAILLMService(
        #     model="gpt-4o-mini",
        # )
        # self.llm = OLLamaLLMService(model="smollm:latest")

        logger.info("Initializing GroqLLMService...")
        llm_start = time.perf_counter()

        self.llm = GroqLLMService(
            model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY")
        )

        logger.info(
            f"GroqLLMService initialized in {time.perf_counter() - llm_start:.2f} seconds"
        )

        logger.info("Initializing OpenAITTSService...")

        tts_start = time.perf_counter()

        # self.tts = OpenAITTSService(
        #     base_url="http://localhost:8880/v1",
        #     api_key="not-needed",
        #     model="kokoro",
        #     sample_rate=24000,
        # )

        self.tts = ChatterboxTTSService()
        logger.info(
            f"OpenAITTSService initialized in {time.perf_counter() - tts_start:.2f} seconds"
        )

        logger.info("Initializing ElevenLabsProcessor...")

        eleven_start = time.perf_counter()

        self.elevenlabs_processor = ElevenLabsProcessor(
            params=self.conversation_config,
        )

        logger.info(
            f"ElevenLabsProcessor initialized in {time.perf_counter() - eleven_start:.2f} seconds"
        )

        logger.info("Building context aggregator...")
        context_start = time.perf_counter()
        self.context_aggregator = self._build_context_aggregator(
            [tool.to_schema() for tool in self.tools]
        )
        logger.info(
            f"Context aggregator built in {time.perf_counter() - context_start:.2f} seconds"
        )

        logger.info("Building pipeline...")
        pipeline_start = time.perf_counter()
        self.pipeline = Pipeline(
            [
                self.ws_transport.input(),
                self.stt,
                self.context_aggregator.user(),
                self.elevenlabs_processor,
                self.llm,
                self.tts,
                self.ws_transport.output(),
                self.context_aggregator.assistant(),
            ]
        )
        logger.info(
            f"Pipeline built in {time.perf_counter() - pipeline_start:.2f} seconds"
        )

        logger.info("Building PipelineTask and PipelineRunner...")

        task_start = time.perf_counter()

        self.task = PipelineTask(
            self.pipeline,
            params=PipelineParams(
                enable_metrics=True,
                allow_interruptions=True,
                enable_usage_metrics=True,
            ),
            enable_turn_tracking=True,
            enable_tracing=self.tracing_enabled,
            conversation_id=self.conversation_id,
        )
        self.runner = PipelineRunner()

        logger.info(
            f"PipelineTask and PipelineRunner built in {time.perf_counter() - task_start:.2f} seconds"
        )
        logger.info(
            f"VoiceAgent._setup_pipeline completed in {time.perf_counter() - setup_start:.2f} seconds"
        )

    async def _on_tool_call(self, params: FunctionCallParams):
        function_name = params.function_name
        if function_name not in self._functions:
            logger.error(f"Function {function_name} not found")
            return
        function_call_item = self._functions[function_name]
        try:
            result = await function_call_item.handler(params.arguments)
            await params.result_callback(result)
        except Exception as e:
            logger.error(f"Error calling function {function_name}: {e}")
            await params.result_callback(str(e))

    def _register_tools(self):
        for tool in self.tools:
            self._functions[tool.name] = FunctionCallRegistryItem(
                function_name=tool.name,
                handler=tool.execute,
                cancel_on_interruption=True,
            )
            self.llm.register_function(tool.name, self._on_tool_call)

    def _register_event_handlers(self):
        @self.ws_transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Pipecat Client connected")
            if self.conversation_config.agent.first_message:
                await self.task.queue_frames(
                    [self.context_aggregator.user().get_context_frame()]
                )

        @self.ws_transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Pipecat Client disconnected")
            await self.task.cancel()

        @self.ws_transport.event_handler("on_session_timeout")
        async def on_session_timeout(transport, client):
            logger.info(f"Entering in timeout for {client.remote_address}")
            await self.task.cancel()

        # Add debug event handlers to track speech events
        @self.task.event_handler("on_frame_reached_downstream")
        async def on_frame_reached_downstream(task, frame):
            if hasattr(frame, "__class__"):
                frame_type = frame.__class__.__name__
                if "Speaking" in frame_type or "Transcription" in frame_type:
                    logger.warning(
                        f"SPEECH DEBUG: Downstream frame: {frame_type} at {time.perf_counter():.2f}"
                    )

        # # Set filter to only track speech-related frames
        # self.task.set_reached_downstream_filter(
        #     (
        #         UserStartedSpeakingFrame,
        #         UserStoppedSpeakingFrame,
        #         TranscriptionFrame,
        #         InterimTranscriptionFrame,
        #         BotStartedSpeakingFrame,
        #         BotStoppedSpeakingFrame,
        #     )
        # )

    async def run(self):
        run_start = time.perf_counter()
        try:
            logger.info("Running _setup_pipeline...")
            setup_start = time.perf_counter()
            self._setup_pipeline()
            logger.info(
                f"_setup_pipeline completed in {time.perf_counter() - setup_start:.2f} seconds"
            )

            logger.info("Registering tools...")
            reg_tools_start = time.perf_counter()
            self._register_tools()
            logger.info(
                f"Tools registered in {time.perf_counter() - reg_tools_start:.2f} seconds"
            )

            logger.info("Registering event handlers...")
            reg_events_start = time.perf_counter()
            self._register_event_handlers()
            logger.info(
                f"Event handlers registered in {time.perf_counter() - reg_events_start:.2f} seconds"
            )

            logger.info("Running pipeline runner...")
            runner_start = time.perf_counter()
            await self.runner.run(self.task)
            logger.info(
                f"Pipeline runner finished in {time.perf_counter() - runner_start:.2f} seconds"
            )
        except Exception as e:
            logger.error(f"Error running voice agent: {e}", e.stack_info)
            raise e
        finally:
            logger.info(
                f"VoiceAgent.run completed in {time.perf_counter() - run_start:.2f} seconds"
            )
