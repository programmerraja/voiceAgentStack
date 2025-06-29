import os
import json
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from pipecat.services.ollama.llm import OLLamaLLMService

# from pipecat.services.fish.tts import FishAudioTTSService
# from pipecat.services.xtts.tts import XTTSService
from pipecat.transcriptions.language import Language
# from service.Dia.tts import DiaTTSService

from serializers.elevenlabs import ElevenLabsFrameSerializer
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from service.whisper.stt import WhisperSTTService
from pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
import aiohttp
from pipecat.services.openai.tts import OpenAITTSService
from dotenv import load_dotenv
from service.Kokoro.tts import KokoroTTSService
# from service.orpheus.tts import OrpheusTTSService


# from service.chatterbot.tts import ChatterboxTTSService

from pipecat.utils.tracing.setup import setup_tracing
from tools.tools import json_to_tools_schema


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


config = json.load(open("config.json"))

async def run_elvenlabs_bot(websocket_client):
    ws_transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=ElevenLabsFrameSerializer(params=ElevenLabsFrameSerializer.InputParams(audio_format="ulaw",sample_rate=8000)),
        ),
    )

    stt = WhisperSTTService(
        model="tiny",
        device="cpu",
        compute_type="default",
        language="en",
    )

    llm = OLLamaLLMService(
        model="smollm:latest",
        # params=OLLamaLLMService.InputParams(temperature=0.7, max_tokens=1000),
    )
    # TODO get prompt from db and put here and need to initito 11lasb processor  

    # Load tools from config/tools.json
    with open("config/tools.json") as f:
        tools_json = json.load(f)
    tools_schema = json_to_tools_schema(tools_json)

    context = OpenAILLMContext(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Start by greeting the user warmly and introducing yourself.",
            },
        ],
        tools=tools_schema
    )
    context_aggregator = llm.create_context_aggregator(context)

   

    TTS = OpenAITTSService(
        base_url="http://localhost:8880/v1",
        api_key="not-needed",
        model="kokoro",
        sample_rate=24000,
    )

   
    pipeline = Pipeline(
        [
            ws_transport.input(),
            stt, 
            context_aggregator.user(),
            llm,
            TTS,  
            ws_transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            allow_interruptions=True,
            enable_usage_metrics=True,
        ),
        enable_turn_tracking=True,
        enable_tracing=IS_TRACING_ENABLED,
        conversation_id="voice-agent-conversation-1",
    )

 

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    @ws_transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport, client):
        logger.info(f"Entering in timeout for {client.remote_address}")
        await task.cancel()

    runner = PipelineRunner()

    await runner.run(task)
