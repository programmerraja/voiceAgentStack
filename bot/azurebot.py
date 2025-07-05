import os

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter



from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.services.azure.stt import AzureSTTService
from pipecat.services.azure.tts import AzureTTSService
from pipecat.services.azure.llm import AzureLLMService
from dotenv import load_dotenv

from pipecat.utils.tracing.setup import setup_tracing
from serializers.elevenlabs import ElevenLabsFrameSerializer

SYSTEM_INSTRUCTION = f"""
"You are Gemini Chatbot, a friendly, helpful robot.

Your goal is to demonstrate your capabilities in a succinct way.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""

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


async def run_bot_websocket_server(websocket_client):
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

    stt = AzureSTTService(
        api_key=os.getenv("AZURE_API_KEY"),
        region=os.getenv("AZURE_REGION"),
        language="en",
        sample_rate=24000,
    )

    llm = AzureLLMService(
        api_key=os.getenv("AZURE_API_KEY"),
        region=os.getenv("AZURE_REGION"),
        model="gpt-4o",
        sample_rate=24000,
    )

    context = OpenAILLMContext(
        [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            },
            {
                "role": "user",
                "content": "Start by greeting the user warmly and introducing yourself.",
            },
        ],
    )
    context_aggregator = llm.create_context_aggregator(context)

    TTS = AzureTTSService(
        api_key=os.getenv("AZURE_API_KEY"),
        region=os.getenv("AZURE_REGION"),
        voice_id="en-US-AriaNeural",
        sample_rate=24000,
    )
    
    pipeline = Pipeline(
        [
            ws_transport.input(),
            stt,  # STT
            context_aggregator.user(),
            llm,
            TTS,  # TTS
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
        # enable_turn_tracking=True,
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
