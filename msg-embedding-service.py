import logging
import sys
import time
from flask import Flask, request, jsonify
from llama_cpp import Llama

LOG_FILE = "msg-embedding-service.log"

logger = logging.getLogger("embedding-service")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

app = Flask(__name__)

MODEL_PATH = "embeddinggemma-300M-Q8_0.gguf"


def load_model():
    logger.info("Loading model: %s", MODEL_PATH)
    t0 = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        embedding=True,
        n_ctx=512,
        verbose=False,
    )
    elapsed = time.time() - t0
    logger.info("Model loaded in %.1fs", elapsed)
    return llm


logger.info("Starting embedding service...")
llm = load_model()


@app.route("/embed", methods=["POST"])
def embed():
    t0 = time.time()
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        if not text:
            logger.warning("Request with empty/missing 'text' field")
            return jsonify({"error": "missing 'text' field"}), 400

        logger.debug("Embedding text (%d chars): %.50s...", len(text), text)

        result = llm.create_embedding(text)
        embedding = result["data"][0]["embedding"]
        elapsed = time.time() - t0

        logger.info(
            "Embedded %d chars -> %d dims in %.0fms",
            len(text), len(embedding), elapsed * 1000,
        )

        return jsonify({
            "text": text,
            "embedding": embedding,
            "dimension": len(embedding),
        })

    except Exception:
        logger.exception("Embedding failed after %.2fs", time.time() - t0)
        return jsonify({"error": "internal server error"}), 500


if __name__ == "__main__":
    logger.info("Listening on :10911")
    app.run(host="0.0.0.0", port=10911, debug=False)
