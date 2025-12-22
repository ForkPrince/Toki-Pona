from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from flask import Flask, request, jsonify
from waitress import serve

app = Flask(__name__)

v1_tok = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
v1_model = AutoModelForSeq2SeqLM.from_pretrained("Jayyydyyy/m2m100_418m_tokipona")

v2_tok = AutoTokenizer.from_pretrained("gpt2")
v2_model = AutoModelForCausalLM.from_pretrained("adamthekiwi/toki-pona-gpt2-alpaca-best")
v2_pipe = pipeline("text-generation", model=v2_model, tokenizer=v2_tok)

@app.route("/v1", methods=["POST"])
def v1_translate():
    try:
        data = request.json
        text, src, target = data["text"], data["src"], data["target"]
        candidates = int(data.get("candidates", 3))

        v1_tok.src_lang = src
        v1_tok.tgt_lang = target
        ins = v1_tok(text, return_tensors="pt")

        outs = v1_model.generate(
            **ins,
            num_beams=candidates,
            num_return_sequences=candidates,
            forced_bos_token_id=v1_tok.lang_code_to_id[target]
        )

        return jsonify({"translation": v1_tok.batch_decode(outs, skip_special_tokens=True)})
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/v2", methods=["POST"])
def v2_translate():
    try:
        data = request.json
        text, direction = data["text"], data["direction"]

        task = "Toki Pona text into English" if direction == "to_english" else "English text into Toki Pona"
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\nTranslate the following {task}:\n{text}\n\n"

        result = v2_pipe(prompt, max_new_tokens=50, do_sample=False, pad_token_id=v2_tok.eos_token_id, return_full_text=False)
        translation = result[0]['generated_text'].split('Below is an instruction')[0].strip()

        return jsonify({"translation": translation})
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/")
def root():
    return jsonify({
        "v1": "M2M100 - multi-language (with toki pona support)",
        "v2": "GPT-2 - Toki Pona <-> English"
    })

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
