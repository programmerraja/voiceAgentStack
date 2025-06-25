# OpenTelemetry Tracing Setup

This voice agent stack now supports OpenTelemetry tracing for monitoring and observability.

## Configuration

### Environment Variables

Add these variables to your `.env` file:

```bash
# Enable/disable tracing
ENABLE_TRACING=true

# Enable console export for debugging
OTEL_CONSOLE_EXPORT=true

# OTLP endpoint (default: http://localhost:4318)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# Optional headers for authentication
OTEL_EXPORTER_OTLP_HEADERS=

# Service name for tracing
OTEL_SERVICE_NAME=voice-agent-stack
```

### Running with Tracing

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your environment**:
   ```bash
   cp env.example .env
   # Edit .env and set ENABLE_TRACING=true
   ```

3. **Run a tracing backend** (optional for local development):
   ```bash
   # Using Jaeger (example)
   docker run -d \
     --name jaeger \
     -p 16686:16686 \
     -p 14250:14250 \
     -p 14268:14268 \
     -p 4317:4317 \
     -p 4318:4318 \
     jaegertracing/all-in-one:latest
   ```

4. **Start the voice agent**:
   ```bash
   python server.py
   ```

## What Gets Traced

When tracing is enabled, the following components are tracked:

- **WebSocket connections** and lifecycle events
- **Speech-to-Text (STT)** processing
- **Language Model (LLM)** interactions
- **Text-to-Speech (TTS)** processing
- **Pipeline execution** and frame processing
- **Context aggregation** and conversation flow

## Viewing Traces

- **Console Output**: Set `OTEL_CONSOLE_EXPORT=true` to see traces in your terminal
- **Jaeger UI**: Visit http://localhost:16686 if using Jaeger
- **Other Backends**: Configure `OTEL_EXPORTER_OTLP_ENDPOINT` to point to your observability platform

## Troubleshooting

### Common Issues

1. **Tracing not working**:
   - Ensure `ENABLE_TRACING=true` in your `.env` file
   - Check that OpenTelemetry dependencies are installed
   - Verify your OTLP endpoint is accessible

2. **Performance impact**:
   - Tracing adds minimal overhead but can be disabled in production
   - Use sampling to reduce trace volume in high-traffic scenarios

3. **Connection issues**:
   - Verify your tracing backend is running and accessible
   - Check firewall settings for the OTLP endpoint

### Debug Mode

For debugging, enable console export:
```bash
OTEL_CONSOLE_EXPORT=true
```

This will print traces directly to your terminal, useful for development and troubleshooting. 