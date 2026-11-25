"""
OpenAI-compatible image generation server for NVIDIA Nemotron-Labs-Image-30B-A3B.

Exposes:
  GET  /v1/models              Health / model discovery check
  POST /v1/images/generations  Generates one PNG image via vLLM + L24SQ decoder

Run with serve.sh (see README.md).
"""

import base64
import io
import os
import re
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_ID = "nvidia/Nemotron-Labs-Image-30B-A3B"
PATCH_SIZE = 16
SYSTEM_PROMPT = (
    "Generate an image matching the requested resolution by detailing the color, "
    "quantity, text, shape, size, texture, spatial relationships of the objects "
    "and background."
)


def _default_tp_size() -> int:
    count = torch.cuda.device_count()
    for tp in (4, 2, 1):
        if count >= tp:
            return tp
    return 1


def _parse_size(size: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", size.strip())
    if not match:
        raise ValueError(
            f"Invalid size '{size}': expected WIDTHxHEIGHT (e.g. 1248x832)"
        )
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0 or width % PATCH_SIZE != 0 or height % PATCH_SIZE != 0:
        raise ValueError(
            f"Width and height must be positive multiples of {PATCH_SIZE}, got {size}"
        )
    return width, height


def extract_l24sq_codes(
    token_ids: list[int], tokenizer, width: int, height: int
) -> tuple[list[int], tuple[int, int]]:
    """Validate one generated image block and return codebook indices."""
    image_start = tokenizer.convert_tokens_to_ids("<|img_start|>")
    image_end = tokenizer.convert_tokens_to_ids("<|img_end|>")
    visual_row = tokenizer.convert_tokens_to_ids("<|vis_row|>")
    vision = tokenizer.unitron_token_ranges["vision"]
    vision_start = int(vision["start_id"])
    vision_end = int(vision["end_id"])

    try:
        start = token_ids.index(image_start)
        end = token_ids.index(image_end, start + 1)
    except ValueError as exc:
        raise RuntimeError("Generation did not contain a complete image block") from exc

    prefix_ids = []
    rows = []
    current_row = None
    for token_id in token_ids[start + 1 : end]:
        if token_id == visual_row:
            if current_row is not None:
                rows.append(current_row)
            current_row = []
        elif vision_start <= token_id < vision_end:
            if current_row is None:
                raise RuntimeError("Visual token appeared before the first row marker")
            current_row.append(token_id - vision_start)
        elif current_row is None:
            prefix_ids.append(token_id)
        else:
            raise RuntimeError("Non-visual token appeared inside the image grid")
    if current_row is not None:
        rows.append(current_row)

    size_header = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    match = re.search(r"height=(\d+)\s+width=(\d+)", size_header)
    if match is None:
        raise RuntimeError("Generation did not contain a valid size header")
    generated_height, generated_width = map(int, match.groups())
    if (generated_height, generated_width) != (height, width):
        raise RuntimeError(
            f"Generated size {(generated_width, generated_height)} does not match "
            f"the requested size {(width, height)}"
        )

    expected_grid = (height // PATCH_SIZE, width // PATCH_SIZE)
    if len(rows) != expected_grid[0] or any(
        len(row) != expected_grid[1] for row in rows
    ):
        raise RuntimeError("Generated visual-token grid has the wrong shape")

    codes = [code for row in rows for code in row]
    return codes, expected_grid


class ImageGenerationRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str
    n: int = 1
    size: str = "1248x832"
    seed: int | None = 42
    temperature: float = 0.9


def _default_gpu_memory_utilization(tp_size: int) -> float:
    # Weights are ~65 GB in bfloat16; on 80 GB total setups (1x80GB or 2x40GB)
    # 0.70 is too low to fit weights + KV cache, while leaving room for the
    # ~1.3 GB L24SQ decoder on cuda:0.
    total_gib = sum(
        torch.cuda.get_device_properties(i).total_memory / (1024**3)
        for i in range(min(tp_size, torch.cuda.device_count()))
    )
    return 0.88 if total_gib < 120 else 0.70


class NemotronImageEngine:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir.resolve()
        self.decoder_dir = self.model_dir / "Nemotron-Labs-Image-Tokenizer"
        self.lock = threading.Lock()

        tp_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", _default_tp_size()))
        default_gpu_mem = _default_gpu_memory_utilization(tp_size)
        gpu_mem = float(os.environ.get("GPU_MEMORY_UTILIZATION", default_gpu_mem))
        max_model_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
        decoder_device = os.environ.get("DECODER_DEVICE", "cuda:0")
        self.decoder_device = decoder_device

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
        )
        self.image_end_id = self.tokenizer.convert_tokens_to_ids("<|img_end|>")

        self.generator = LLM(
            model=str(self.model_dir),
            tokenizer=str(self.model_dir),
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            moe_backend="triton",
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem,
            enforce_eager=True,
            seed=42,
        )

        self.decoder = (
            AutoModel.from_pretrained(
                str(self.decoder_dir),
                trust_remote_code=True,
                dtype=torch.bfloat16,
            )
            .to(decoder_device)
            .eval()
        )

    def generate(
        self, prompt: str, size: str, seed: int | None, temperature: float
    ) -> tuple[bytes, int, int]:
        width, height = _parse_size(size)
        grid_tokens = (width // PATCH_SIZE) * (height // PATCH_SIZE)
        # Allow visual tokens + row markers + size header + framing tokens.
        max_tokens = max(4352, grid_tokens + (height // PATCH_SIZE) + 128)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Generate a {width}x{height} image.\n\n{prompt}",
            },
        ]
        prompt_token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors=None,
        )

        sampling = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            seed=seed,
            n=1,
            skip_special_tokens=False,
            detokenize=True,
            stop=["<|img_end|>"],
            stop_token_ids=[self.image_end_id],
            include_stop_str_in_output=True,
        )

        with self.lock:
            result = self.generator.generate(
                [{"prompt_token_ids": prompt_token_ids}],
                sampling,
                use_tqdm=False,
            )[0]
            generated_token_ids = list(result.outputs[0].token_ids)
            codes, grid_size = extract_l24sq_codes(
                generated_token_ids, self.tokenizer, width, height
            )

            indices = torch.tensor(codes, dtype=torch.long, device=self.decoder_device)
            with torch.inference_mode():
                decoded = self.decoder.pack_decode(
                    indices_list=[indices],
                    grid_sizes=[grid_size],
                )[0]

            pixels = (
                ((decoded.float().clamp(-1, 1) + 1) / 2 * 255)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="PNG")
        return buf.getvalue(), len(prompt_token_ids), len(generated_token_ids)


def create_app(engine: NemotronImageEngine) -> FastAPI:
    app = FastAPI(title="Nemotron-Labs-Image-30B-A3B Server")

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model"}],
        }

    @app.post("/v1/images/generations")
    def generate_image(req: ImageGenerationRequest):
        if req.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is supported.")
        try:
            png_bytes, input_tokens, output_tokens = engine.generate(
                prompt=req.prompt,
                size=req.size,
                seed=req.seed,
                temperature=req.temperature,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(png_bytes).decode("ascii")}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

    return app


def main() -> None:
    model_dir = Path(
        os.environ.get("MODEL_DIR", str(Path.home() / "Nemotron-Labs-Image-30B-A3B"))
    )
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8001"))

    engine = NemotronImageEngine(model_dir)
    app = create_app(engine)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
