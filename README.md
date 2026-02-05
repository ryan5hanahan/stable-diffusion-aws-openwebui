# Stable Diffusion AWS Bedrock + OpenWebUI

A Docker Compose setup that integrates OpenWebUI with AWS Bedrock's Stability AI models for image generation.

## Architecture

- **OpenWebUI**: Web interface for interacting with AI models (port 3000)
- **Bedrock Proxy**: FastAPI service that translates OpenWebUI requests to AWS Bedrock Stability AI API calls (port 8000)

The proxy provides OpenAI-compatible chat completions and image generation endpoints, allowing OpenWebUI to generate images using AWS Bedrock.

## Prerequisites

- Docker and Docker Compose
- AWS account with Bedrock access
- AWS credentials with permissions for `bedrock:InvokeModel`
- Stability AI model access enabled in AWS Bedrock console

### Enable Bedrock Model Access

1. Go to AWS Console → Bedrock → Model access
2. Request access to Stability AI models (SD 3.5 Large, Stable Image Core, or Stable Image Ultra)
3. Wait for approval (usually instant)

**Important**: Stability AI models are only available in **us-west-2**.

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/stable-diffusion-aws-openwebui.git
   cd stable-diffusion-aws-openwebui
   ```

2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

3. Edit `.env` with your AWS credentials:
   ```bash
   AWS_REGION=us-west-2
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   SD_MODEL_ID=stability.sd3-5-large-v1:0
   ```

4. Start the services:
   ```bash
   docker-compose up -d
   ```

5. Access OpenWebUI at http://localhost:3000

## Available Models

| Model ID | Description | Best For |
|----------|-------------|----------|
| `stability.sd3-5-large-v1:0` | Stable Diffusion 3.5 Large | Best quality, recommended |
| `stability.stable-image-core-v1:1` | Stable Image Core | Fast and affordable |
| `stability.stable-image-ultra-v1:1` | Stable Image Ultra | Premium photorealistic |

All models require `AWS_REGION=us-west-2`.

## Usage

### In OpenWebUI

1. Create an account (stored locally)
2. Select a Stability model from the model dropdown (e.g., `stability.stable-image-core-v1:1`)
3. Type your image prompt and send

The proxy treats all chat messages as image generation prompts and returns the generated image.

### Example Prompts

- "A sunset over mountains with vibrant orange and purple colors"
- "A cyberpunk cityscape at night with neon lights"
- "A cute robot reading a book in a cozy library"

### Direct API Usage

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Generate image via chat completions
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "stability.sd3-5-large-v1:0", "messages": [{"role": "user", "content": "a cat wearing sunglasses"}]}'

# Generate image via OpenAI images endpoint
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a mountain landscape", "size": "1024x1024"}'
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List available models (OpenAI-compatible) |
| `POST /v1/chat/completions` | Chat completions that generate images |
| `POST /v1/images/generations` | OpenAI-compatible image generation |
| `POST /sdapi/v1/txt2img` | Automatic1111-compatible text-to-image |
| `GET /sdapi/v1/sd-models` | List models (A1111 format) |

## Supported Parameters

The new Stability AI API supports:
- `prompt` - Image description (required)
- `negative_prompt` - What to exclude from the image
- `aspect_ratio` - One of: 16:9, 1:1, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, 9:21
- `seed` - For reproducible generations (0-4294967294)

**Note**: `cfg_scale` and `steps` parameters are accepted for compatibility but ignored—these are model-controlled in the new API.

## Development

```bash
# View logs
docker-compose logs -f bedrock-image-proxy

# Rebuild after code changes
docker-compose up -d --build bedrock-image-proxy

# Stop services
docker-compose down
```

## Troubleshooting

### "Model not found" errors
- Ensure `AWS_REGION=us-west-2` in your `.env` file
- Stability AI models are only available in us-west-2

### "Access denied" errors
- Verify AWS credentials have `bedrock:InvokeModel` permission
- Check that Stability AI model access is enabled in AWS Console → Bedrock → Model access

### Images not generating
- Check logs: `docker-compose logs bedrock-image-proxy`
- Test the proxy: `curl http://localhost:8000/health`

### Content filtered
- The Stability AI models have content filtering
- Try rephrasing your prompt if it gets blocked

## Cost Considerations

AWS Bedrock charges per image. Check [AWS Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) for current rates.

## License

MIT
