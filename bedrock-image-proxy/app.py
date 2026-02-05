"""
AWS Bedrock Stable Diffusion Proxy
Provides OpenAI-compatible API endpoints for image generation using AWS Bedrock

Supports the new Stability AI models (SD 3.5, Stable Image Core, Stable Image Ultra)
which replaced the deprecated SDXL models.
"""
import base64
import json
import os
import time
from typing import Optional

import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Bedrock SD Proxy")

# AWS Bedrock client - New Stability AI models only available in us-west-2
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv('AWS_REGION', 'us-west-2')
)

# Default to Stable Diffusion 3.5 Large - best balance of quality and speed
# Available models:
#   - stability.sd3-5-large-v1:0     (SD 3.5 Large - high quality, high volume)
#   - stability.stable-image-core-v1:1  (Stable Image Core - fast and affordable)
#   - stability.stable-image-ultra-v1:1 (Stable Image Ultra - photorealistic, premium)
SD_MODEL_ID = os.getenv('SD_MODEL_ID', 'stability.sd3-5-large-v1:0')

# Supported aspect ratios for new Stability AI models
SUPPORTED_ASPECT_RATIOS = [
    "16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"
]


class ImageGenerationRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    model: Optional[str] = "dall-e-3"


class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    # Note: steps and cfg_scale are accepted for compatibility but ignored
    # New Stability AI models don't support these parameters
    steps: Optional[int] = 50
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    cfg_scale: Optional[float] = 7.0
    seed: Optional[int] = None


def dimensions_to_aspect_ratio(width: int, height: int) -> str:
    """Convert width/height to nearest supported aspect ratio"""
    target_ratio = width / height

    # Calculate ratios for each supported aspect ratio
    ratio_map = {
        "16:9": 16/9,
        "1:1": 1/1,
        "21:9": 21/9,
        "2:3": 2/3,
        "3:2": 3/2,
        "4:5": 4/5,
        "5:4": 5/4,
        "9:16": 9/16,
        "9:21": 9/21,
    }

    # Find closest match
    closest = "1:1"
    min_diff = float('inf')
    for aspect, ratio in ratio_map.items():
        diff = abs(target_ratio - ratio)
        if diff < min_diff:
            min_diff = diff
            closest = aspect

    return closest


def parse_size(size: str) -> tuple[int, int]:
    """Parse size string like '1024x1024' into width, height"""
    try:
        width, height = size.split('x')
        return int(width), int(height)
    except:
        return 1024, 1024


def generate_image_bedrock(prompt: str, negative_prompt: str = "", width: int = 1024,
                          height: int = 1024, seed: Optional[int] = None) -> tuple[bytes, int]:
    """
    Generate image using AWS Bedrock Stability AI models.

    Note: The new Stability AI API (SD 3.5, Stable Image Core/Ultra) uses a simplified
    request format without cfg_scale or steps parameters. These are model-controlled.

    Returns:
        tuple: (image_bytes, seed_used)
    """
    aspect_ratio = dimensions_to_aspect_ratio(width, height)

    # New Stability AI request format
    request_body = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "output_format": "png",
    }

    if negative_prompt:
        request_body["negative_prompt"] = negative_prompt

    if seed is not None and seed > 0:
        request_body["seed"] = seed

    try:
        response = bedrock_runtime.invoke_model(
            modelId=SD_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )

        response_body = json.loads(response['body'].read())

        # Check for content filtering
        finish_reasons = response_body.get('finish_reasons', [None])
        if finish_reasons and finish_reasons[0] is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Image generation blocked: {finish_reasons[0]}"
            )

        # New API returns images directly as base64 strings in 'images' array
        if 'images' in response_body and len(response_body['images']) > 0:
            image_base64 = response_body['images'][0]
            seed_used = response_body.get('seeds', [-1])[0]
            return base64.b64decode(image_base64), seed_used
        else:
            raise Exception("No images in response")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bedrock error: {str(e)}")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "bedrock-sd-proxy",
        "model": SD_MODEL_ID,
        "region": os.getenv('AWS_REGION', 'us-west-2'),
        "supported_aspect_ratios": SUPPORTED_ASPECT_RATIOS
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model": SD_MODEL_ID}


@app.post("/v1/images/generations")
async def create_image_openai_compatible(request: ImageGenerationRequest):
    """OpenAI-compatible image generation endpoint"""

    width, height = parse_size(request.size)

    try:
        image_bytes, seed_used = generate_image_bedrock(
            prompt=request.prompt,
            width=width,
            height=height
        )

        # Convert to base64 for response
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        return {
            "created": int(time.time()),
            "data": [
                {
                    "url": f"data:image/png;base64,{image_base64}",
                    "b64_json": image_base64 if request.response_format == "b64_json" else None
                }
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sdapi/v1/txt2img")
async def txt2img(request: TextToImageRequest):
    """
    Automatic1111-compatible text-to-image endpoint for OpenWebUI.

    Note: The steps and cfg_scale parameters are accepted for compatibility
    but are ignored - the new Stability AI models don't support them.
    Width/height are converted to the nearest supported aspect ratio.
    """

    try:
        image_bytes, seed_used = generate_image_bedrock(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            width=request.width,
            height=request.height,
            seed=request.seed
        )

        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Calculate aspect ratio for response info
        aspect_ratio = dimensions_to_aspect_ratio(request.width, request.height)

        # Return in Automatic1111 format
        return {
            "images": [image_base64],
            "parameters": {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "steps": request.steps,  # Echo back for compatibility
                "width": request.width,
                "height": request.height,
                "cfg_scale": request.cfg_scale,  # Echo back for compatibility
                "seed": seed_used
            },
            "info": json.dumps({
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "model": SD_MODEL_ID,
                "aspect_ratio": aspect_ratio,
                "seed": seed_used,
                "note": "steps and cfg_scale are not supported by new Stability AI models"
            })
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sdapi/v1/sd-models")
async def get_models():
    """
    Return available models (Automatic1111 compatibility).

    Lists all supported Stability AI models on AWS Bedrock.
    """
    available_models = [
        {
            "title": "Stable Diffusion 3.5 Large",
            "model_name": "stability.sd3-5-large-v1:0",
            "hash": "bedrock-sd35",
            "sha256": "bedrock-sd35",
            "filename": "stability.sd3-5-large-v1:0",
            "config": None
        },
        {
            "title": "Stable Image Core v1.1",
            "model_name": "stability.stable-image-core-v1:1",
            "hash": "bedrock-core",
            "sha256": "bedrock-core",
            "filename": "stability.stable-image-core-v1:1",
            "config": None
        },
        {
            "title": "Stable Image Ultra v1.1",
            "model_name": "stability.stable-image-ultra-v1:1",
            "hash": "bedrock-ultra",
            "sha256": "bedrock-ultra",
            "filename": "stability.stable-image-ultra-v1:1",
            "config": None
        },
    ]

    # Put currently selected model first
    available_models.sort(key=lambda m: m["model_name"] != SD_MODEL_ID)

    return available_models


@app.get("/sdapi/v1/options")
async def get_options():
    """Return API options (Automatic1111 compatibility)"""
    return {
        "sd_model_checkpoint": SD_MODEL_ID,
        "samples_save": True,
        "samples_format": "png",
        "CLIP_stop_at_last_layers": 1,  # Compatibility field
    }


@app.get("/sdapi/v1/samplers")
async def get_samplers():
    """Return available samplers (Automatic1111 compatibility)"""
    # New Stability AI models don't expose sampler selection
    # Return a default for compatibility
    return [
        {"name": "Default", "aliases": ["default"], "options": {}},
    ]


@app.get("/sdapi/v1/upscalers")
async def get_upscalers():
    """Return available upscalers (Automatic1111 compatibility)"""
    return [
        {"name": "None", "model_name": None, "model_path": None, "model_url": None, "scale": 1},
    ]
