
from flask import Flask, request, jsonify
from flask_cors import CORS
from gemini_service import generate_response
from claim_processor import process_gemini_response

app = Flask(__name__)
CORS(app)



@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        response = generate_response(prompt)
        claims = process_gemini_response(response)

        return jsonify({
            "gemini_response": response,
            "claims": [
                {"claim_id": f"C{i}", "claim": claim}
                for i, claim in enumerate(claims, start=1)
            ]
        }), 200

    except RuntimeError as e:
        error_message = str(e)

        if "temporarily overloaded" in error_message.lower():
            return jsonify({
                "error": error_message
            }), 503

        app.logger.exception("Runtime error during generation")
        return jsonify({
            "error": "An error occurred while generating the response."
        }), 500

    except Exception:
        app.logger.exception("Error while generating or extracting claims")
        return jsonify({
            "error": "An unexpected error occurred. Please try again later."
        }), 500

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({
            "error": "Prompt is required"
        }), 400

    try:
        # Generate the Gemini response
        response = generate_response(prompt)

        # Extract candidate claims from the response
        claims = process_gemini_response(response)

        # Return both the response and extracted claims
        return jsonify({
            "gemini_response": response,
            "claims": [
                {
                    "claim_id": f"C{i}",
                    "claim": claim
                }
                for i, claim in enumerate(claims, start=1)
            ]
        })

    
    except Exception as e:
        app.logger.exception("Error while generating or extracting claims")

        error_text = str(e)

        if "503" in error_text or "UNAVAILABLE" in error_text:
            return jsonify({
                "error": (
                    "Gemini 3.8 Flash is temporarily unavailable. "
                    "Please wait a little and try again."
                )
            }), 503

        return jsonify({
            "error": "An error occurred while generating or extracting claims."
        }), 500



if __name__ == "__main__":
    app.run(debug=True)
