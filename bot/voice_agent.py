import os
import json
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from pipecat.services.openai.llm import OpenAILLMService


from frameworks.elevenlabs import (
    ElevenLabsProcessor,
    ConversationConfig,
)
from serializers.elevenlabs import ElevenLabsFrameSerializer
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from service.whisper.stt import WhisperSTTService

from pipecat.services.openai.tts import OpenAITTSService
from dotenv import load_dotenv


from pipecat.utils.tracing.setup import setup_tracing
from tools.tools import build_tools, json_to_tools_schema, WebhookTool
from pipecat.adapters.schemas.function_schema import FunctionSchema
from typing import Dict, Optional, Any
from pipecat.services.llm_service import FunctionCallRegistryItem, FunctionCallParams
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.ollama.llm import OLLamaLLMService

load_dotenv(override=True)

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
        self.websocket_client = websocket_client
       
        self.tools_config_path = os.path.join(os.path.dirname(__file__), "../config/tools.json")
        self.conversation_config_path = os.path.join(os.path.dirname(__file__), "../config/conversation.json")
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
        self._load_conversation_config()
        
        
    def _load_tools(self):
        with open(self.tools_config_path) as f:
            self.tools_json = json.load(f)
        self.tools = build_tools(self.tools_json)
        
    def _load_conversation_config(self):
        with open(self.conversation_config_path) as f:
            config = json.load(f)
            self.conversation_config = ConversationConfig(**config["conversation_config"])
            self.context_messages.append({"role": "system", "content": self.conversation_config.agent.prompt.prompt})

    def _build_context_aggregator(self, tools_schema):
        context = OpenAILLMContext(self.context_messages, tools=ToolsSchema(standard_tools=tools_schema))
        return self.llm.create_context_aggregator(context)

    def _setup_pipeline(self):
       
        self.ws_transport = FastAPIWebsocketTransport(
            websocket=self.websocket_client,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_analyzer=SileroVADAnalyzer(),
                serializer=ElevenLabsFrameSerializer(
                    params=ElevenLabsFrameSerializer.InputParams(
                        audio_format="ulaw", sample_rate=8000
                    )
                ),
            ),
        )
        
        self.stt = WhisperSTTService(
            model="tiny",
            device="cpu",
            compute_type="default",
            language="en",
        )
        # self.llm = OpenAILLMService(
        #     model="gpt-4o-mini",
        # )
        self.llm = OLLamaLLMService(model="smollm:latest")
        
        self.tts = OpenAITTSService(
            base_url="http://localhost:8880/v1",
            api_key="not-needed",
            model="kokoro",
            sample_rate=24000,
        )
        self.elevenlabs_processor = ElevenLabsProcessor(
            params=self.conversation_config,
        )
        
        self.context_aggregator = self._build_context_aggregator(
            [tool.to_schema() for tool in self.tools]
        )

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
            print("first message",self.conversation_config.agent.first_message,self.context_aggregator.user().get_context_frame())
            if self.conversation_config.agent.first_message:
                await self.task.queue_frames([self.context_aggregator.user().get_context_frame()])
           

        @self.ws_transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Pipecat Client disconnected")
            await self.task.cancel()

        @self.ws_transport.event_handler("on_session_timeout")
        async def on_session_timeout(transport, client):
            logger.info(f"Entering in timeout for {client.remote_address}")
            await self.task.cancel()

    async def run(self):
        try:
            self._setup_pipeline()
            self._register_tools()
            self._register_event_handlers()
            await self.runner.run(self.task)
            
        
        except Exception as e:
            logger.error(f"Error running voice agent: {e}",e.stack_info)
            raise e
