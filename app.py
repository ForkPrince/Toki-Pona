from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, cast

from flask import Flask, Response, jsonify, request
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GPT2Tokenizer,
    M2M100Tokenizer,
    PreTrainedModel,
    TextGenerationPipeline,
    pipeline,
)
from waitress import serve

app = Flask(__name__)

@app.after_request
def add_cors_headers(resp: Response) -> Response:
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    return resp

_v1_lock = threading.Lock()
_v2_lock = threading.Lock()

@dataclass
class V1Resources:
    tok: M2M100Tokenizer
    model: PreTrainedModel

@dataclass
class V2Resources:
    tok: GPT2Tokenizer
    pipe: TextGenerationPipeline

_v1: V1Resources | None = None
_v2: V2Resources | None = None

def get_v1() -> V1Resources:
    global _v1
    if _v1 is None:
        tok = cast(M2M100Tokenizer, AutoTokenizer.from_pretrained("facebook/m2m100_418M"))
        model = AutoModelForSeq2SeqLM.from_pretrained("Jayyydyyy/m2m100_418m_tokipona")
        _v1 = V1Resources(tok=tok, model=model)
    return _v1

def get_v2() -> V2Resources:
    global _v2
    if _v2 is None:
        tok = cast(GPT2Tokenizer, AutoTokenizer.from_pretrained("gpt2"))
        model = AutoModelForCausalLM.from_pretrained("adamthekiwi/toki-pona-gpt2-alpaca-best")
        pipe = pipeline("text-generation", model=model, tokenizer=tok)
        _v2 = V2Resources(tok=tok, pipe=pipe)
    return _v2

def get_json_fields(*fields: str) -> tuple[dict[str, Any], None] | tuple[None, tuple[Response, int]]:
    """Parse JSON body and check required fields; returns (data, error_response)."""
    data = request.get_json(silent=True)
    if not data:
        return None, (jsonify(error="Request body must be JSON"), 400)
    if not all(data.get(f) for f in fields):
        return None, (jsonify(error=f"Missing required fields: {', '.join(fields)}"), 400)
    return data, None

@app.route("/v1", methods=["POST"])
def v1_translate():
    data, err = get_json_fields("text", "src", "target")
    if err:
        return err
    assert data is not None

    text, src, target = data["text"], data["src"], data["target"]
    candidates = int(data.get("candidates", 3))

    with _v1_lock:
        res = get_v1()
        res.tok.src_lang = src
        res.tok.tgt_lang = target
        inputs = res.tok(text, return_tensors="pt")
        outs = cast(Any, res.model).generate(
            **inputs,
            num_beams=candidates,
            num_return_sequences=candidates,
            forced_bos_token_id=res.tok.lang_code_to_id[target],
        )
        translations = res.tok.batch_decode(outs, skip_special_tokens=True)

    return jsonify({"translation": translations})

@app.route("/v2", methods=["POST"])
def v2_translate():
    data, err = get_json_fields("text", "direction")
    if err:
        return err
    assert data is not None

    text, direction = data["text"], data["direction"]
    if direction not in ("to_english", "to_toki_pona"):
        return jsonify(error="Invalid direction"), 400

    task = "Toki Pona text into English" if direction == "to_english" else "English text into Toki Pona"
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"Translate the following {task}:\n{text}\n\n"
    )

    with _v2_lock:
        res = get_v2()
        result = res.pipe(
            prompt,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=res.tok.eos_token_id,
            return_full_text=False,
        )

    translation = result[0]["generated_text"]
    translation = translation.split("Below is an instruction")[0].strip()
    return jsonify({"translation": translation})

@app.route("/")
def root():
    return jsonify({
        "v1": "M2M100 - multi-language (with toki pona support)",
        "v2": "GPT-2 - Toki Pona <-> English",
    })

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
