from flask import Flask, request, jsonify
from gemini_service import generate_response

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    prompt = data.get("prompt")

    if not prompt:
        return jsonify({
            "error": "Prompt is required"
        }), 400

    try:
        response = generate_response(prompt)

        return jsonify({
            "gemini_response": response
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
