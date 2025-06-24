import os

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

from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
import aiohttp

from dotenv import load_dotenv
from service.Kokoro.tts import KokoroTTSService
# from service.orpheus.tts import OrpheusTTSService

# from service.chatterbot.tts import ChatterboxTTSService

from pipecat.utils.tracing.setup import setup_tracing

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
            serializer=ProtobufFrameSerializer(),
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

    # TTS = FishAudioTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY"),
    #     voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Reading Lady
    # )
    # async with aiohttp.ClientSession() as session:
    #     TTS = XTTSService(
    #         voice_id="speaker_1",
    #         language=Language.EN,
    #         base_url="http://localhost:8000",
    #         aiohttp_session=session
    #     )

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

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    TTS = KokoroTTSService(
        model_path=os.path.join(
            os.path.dirname(__file__), "assets", "kokoro-v1.0.fp16-gpu.onnx"
        ),
        voices_path=os.path.join(os.path.dirname(__file__), "assets", "voices.json"),
        voice_id="af",
        sample_rate=16000,
    )

    # TTS = OrpheusTTSService(
    #     model_name="canopylabs/orpheus-3b-0.1-ft",
    #     sample_rate=16000,
    # )

    # TTS = ChatterboxTTSService(
    #     model_name="",
    #     sample_rate=16000,
    # )

    # TTS = DiaTTSService(
    #     model_name="nari-labs/Dia-1.6B",
    #     sample_rate=16000,
    # )
    pipeline = Pipeline(
        [
            ws_transport.input(),
            rtvi,
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
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()
        # Kick off the conversation.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

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
