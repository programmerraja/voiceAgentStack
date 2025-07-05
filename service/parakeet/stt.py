import asyncio
import queue
import threading
from pipecat.services.stt_service import STTService
from pipecat.frames.frames import TranscriptionFrame, InterimTranscriptionFrame
from pipecat.audio.vad.silero import SileroVADAnalyzer
import torch
import nemo.collections.asr as nemo_asr

class ParakeetStreamingSTTService(STTService):
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,  # 100ms chunks
        vad_threshold: float = 0.5,
        batch_size: int = 4,
        enable_partial_results: bool = True,
        **kwargs
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._model_name = model_name
        self._chunk_duration_ms = chunk_duration_ms
        self._vad_threshold = vad_threshold
        self._batch_size = batch_size
        self._enable_partial_results = enable_partial_results
        
        # Audio processing
        self._audio_buffer = []
        self._vad_analyzer = SileroVADAnalyzer()
        self._processing_queue = asyncio.Queue()
        self._batch_queue = []
        
        # NeMo model
        self._model = None
        self._is_model_loaded = False
        
        # Background processing
        self._processing_task = None
        self._batch_processing_task = None
        
    async def start(self, frame):
        """Initialize streaming components"""
        await super().start(frame)
        
        # Load NeMo Parakeet model
        if not self._is_model_loaded:
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self._model_name
            )
            
            if torch.cuda.is_available():
                self._model = self._model.cuda().half()
            
            self._model.eval()
            self._is_model_loaded = True
        
        # Start background processing tasks
        self._processing_task = asyncio.create_task(self._process_audio_stream())
        self._batch_processing_task = asyncio.create_task(self._batch_processor())
        
    async def stop(self, frame):
        """Clean up streaming components"""
        if self._processing_task:
            self._processing_task.cancel()
        if self._batch_processing_task:
            self._batch_processing_task.cancel()
            
        await super().stop(frame)
        
    async def run_stt(self, audio: bytes):
        """Queue audio for processing"""
        if not self._is_model_loaded:
            return
            
        # Add to processing queue
        await self._processing_queue.put(audio)
        
    async def _process_audio_stream(self):
        """Process audio stream with VAD"""
        while True:
            try:
                audio_chunk = await self._processing_queue.get()
                
                # Convert to numpy for VAD
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
                audio_float = audio_np.astype(np.float32) / 32768.0
                
                # Check for speech activity
                if self._vad_analyzer.analyze_audio(audio_float):
                    self._audio_buffer.extend(audio_float)
                    
                    # Process when buffer is sufficiently large
                    if len(self._audio_buffer) >= self.sample_rate * 0.5:  # 0.5 seconds
                        await self._queue_for_transcription()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in audio processing: {e}")
                
    async def _queue_for_transcription(self):
        """Queue audio buffer for batch transcription"""
        if self._audio_buffer:
            audio_segment = np.array(self._audio_buffer)
            self._batch_queue.append(audio_segment)
            self._audio_buffer = []
            
            # Process batch when full
            if len(self._batch_queue) >= self._batch_size:
                await self._process_batch()
                
    async def _batch_processor(self):
        """Background batch processing"""
        while True:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms
                
                # Process partial batches to reduce latency
                if self._batch_queue and len(self._batch_queue) >= 1:
                    await self._process_batch()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in batch processing: {e}")
                
    async def _process_batch(self):
        """Process audio batch with Parakeet"""
        if not self._batch_queue:
            return
            
        try:
            batch = self._batch_queue.copy()
            self._batch_queue.clear()
            
            with torch.no_grad():
                # Process each audio segment
                for audio_segment in batch:
                    hypotheses = self._model.transcribe(
                        audio=[audio_segment],
                        timestamps=True,
                        logprobs=False
                    )
                    
                    if hypotheses and len(hypotheses) > 0:
                        hypothesis = hypotheses[0]
                        
                        if hasattr(hypothesis, 'text') and hypothesis.text:
                            # Create transcription frame
                            frame = TranscriptionFrame(
                                text=hypothesis.text,
                                user_id="user",
                                timestamp=asyncio.get_event_loop().time()
                            )
                            
                            # Send to pipeline
                            await self._send_frame(frame)
                            
        except Exception as e:
            print(f"Error in batch transcription: {e}")
            
    async def _send_frame(self, frame):
        """Send frame to pipeline (override based on your pipeline setup)"""
        # This would be implemented based on your specific pipeline architecture
        pass
