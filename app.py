from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
import os
from PIL import Image
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Hugging Face API key from environment variable
api_key = os.environ.get("api_key")
client = InferenceClient(api_key=api_key)

def resize_and_convert_to_base64(image_path, resized_width=420, resized_height=280):
    """
    Resizes a local image file, converts it to base64, and returns it as a data URL.
    """
    try:
        img = Image.open(image_path)

        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img = img.resize((resized_width, resized_height))

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

@app.route("/")
def home():
    """
    Simple endpoint to confirm the backend server is running.
    """
    return "Hello! The backend server is running."

@app.route("/text", methods=["POST"])
def extract_text():
    """
    Extracts text from the provided image file.
    """
    image_path = None
    try:
        # Check if the request contains an image file
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        # Save the uploaded image file to a temporary location
        image = request.files['image']
        image_path = f"temp_{image.filename}"
        image.save(image_path)

        # Resize and convert the image to base64
        resized_image = resize_and_convert_to_base64(image_path)

        # Define the system and user prompt
        system_prompt = {
            "role": "system",
            "content": "You are an expert in extracting text inside given images. Give the text inside the image given by the user."
        }

        messages = [
            system_prompt,
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": resized_image},
                    },
                    {"type": "text", "text": "Give me the text inside the image"}
                ]
            }
        ]

        # Send request to Hugging Face API
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct",
            messages=messages,
            max_tokens=3000
        )

        # Check and return response content
        if response and response.get('choices'):
            return jsonify({"text": response['choices'][0]['message']['content']})
        else:
            return jsonify({"error": "No valid content received in response"}), 500

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    finally:
        # Delete the temporary image file
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
