from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch
import soundfile as sf
import librosa
import numpy as np
from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and feature extractor
try:
    model_name = "./wav2vec2_model"  # Replace with your model path
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
    TARGET_SAMPLE_RATE = 16000
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

emotion_labels = ["fear", "sad", "angry", "surprise", "disgust", "neutral", "happy"]

def load_and_resample_audio(file_path):
    """Load and resample audio to 16kHz."""
    # Load audio with original sampling rate
    audio, orig_sr = librosa.load(file_path, sr=None)
    
    # Resample if necessary
    if orig_sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE)
    
    # Normalize audio
    audio = librosa.util.normalize(audio)
    
    return audio, TARGET_SAMPLE_RATE

def predict_emotion(audio_path):
    try:
        # Load and resample audio
        speech, sr = load_and_resample_audio(audio_path)
        
        # Convert to float32 if needed
        speech = speech.astype(np.float32)
        
        # Extract features
        inputs = feature_extractor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
        
        # Make prediction
        with torch.no_grad():
            logits = model(**inputs).logits
            predictions = torch.nn.functional.softmax(logits, dim=-1)
            
        # Get top 3 predictions
        top_3_probs, top_3_indices = torch.topk(predictions[0], 3)
        
        results = []
        for prob, idx in zip(top_3_probs, top_3_indices):
            results.append({
                'emotion': emotion_labels[idx],
                'probability': float(prob) * 100
            })
        
        return {'status': 'success', 'predictions': results}
    except Exception as e:
        print(f"Error in predict_emotion: {str(e)}")
        traceback.print_exc()
        return {'status': 'error', 'message': str(e)}

@app.route('/')
def home():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Emotion Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f8ff; /* Light blue background */
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            height: 100vh;
        }
        h1 {
            font-size: 2.5em;
            color: #333333;
            text-align: center;
            margin-top: 30px;
        }
        .quote {
            font-size: 1.2em;
            color: #666666;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
            width: 100%;
            max-width: 500px;
        }
        .container h2 {
            font-size: 1.5em;
            color: #333333;
            margin-bottom: 20px;
        }
        form {
            display: flex;
            flex-direction: column;
            gap: 15px;
            align-items: center;
        }
        .form-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        input[type="file"] {
            padding: 10px;
            border: 1px solid #cccccc;
            border-radius: 5px;
            background-color: #f9f9f9;
            cursor: pointer;
            font-size: 1em;
        }
        button {
            padding: 10px 20px;
            font-size: 1em;
            font-weight: bold;
            color: #ffffff;
            background-color: #4CAF50;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        button:hover {
            background-color: #45a049;
            transform: scale(1.1);
        }
        .loading, .error-message {
            margin-top: 15px;
            font-size: 0.9em;
            display: none;
        }
        .loading {
            color: #ff9900;
        }
        .error-message {
            color: #ff4d4d;
        }
        .results {
            margin-top: 20px;
            display: none;
        }
        .results h2 {
            font-size: 1.4em;
            color: #333333;
            margin-bottom: 15px;
        }
        .emotion-bar {
            background-color: #f0f0f0;
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            text-align: left;
            font-size: 1em;
            position: relative;
        }
        .emotion-bar .label {
            font-weight: bold;
            color: #333333;
            margin-bottom: 5px;
        }
        .progress {
            background-color: #4CAF50;
            height: 10px;
            border-radius: 4px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>Audio Emotion Detection</h1>
    <div class="quote">"The voice is a powerful tool; it conveys more than just words."</div>
    <div class="container">
        <h2>Choose the audio for emotion prediction</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <input type="file" name="audio" accept="audio/*" required>
                <button type="submit">Analyze</button>
            </div>
        </form>
        <div id="loading" class="loading">🔄 Processing audio...</div>
        <div id="errorMessage" class="error-message">⚠️ Error message</div>
        <div id="results" class="results">
            <h2>Top 3 Emotions</h2>
            <div id="emotionResults"></div>
        </div>
    </div>
    <script>
        document.getElementById('uploadForm').onsubmit = function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const errorMessage = document.getElementById('errorMessage');
            const results = document.getElementById('results');
            const loading = document.getElementById('loading');

            // Reset display
            errorMessage.style.display = 'none';
            results.style.display = 'none';
            loading.style.display = 'block';

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';

                if (data.status === 'error') {
                    errorMessage.textContent = data.message;
                    errorMessage.style.display = 'block';
                    return;
                }

                const emotionResults = document.getElementById('emotionResults');
                emotionResults.innerHTML = '';

                data.predictions.forEach(result => {
                    const bar = document.createElement('div');
                    bar.className = 'emotion-bar';
                    bar.innerHTML = `
                        <div class="label">${result.emotion}: ${result.probability.toFixed(2)}%</div>
                        <div class="progress" style="width: ${result.probability}%;"></div>
                    `;
                    emotionResults.appendChild(bar);
                });

                results.style.display = 'block';
            })
            .catch(error => {
                loading.style.display = 'none';
                console.error('Error:', error);
                errorMessage.textContent = 'An error occurred while processing the request.';
                errorMessage.style.display = 'block';
            });
        };
    </script>
</body>
</html>

    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No audio file uploaded'
            }), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No selected file'
            }), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            results = predict_emotion(filepath)
            return jsonify(results)
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)