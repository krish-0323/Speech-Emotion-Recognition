# Speech Emotion Recognition

## Purpose
This project aims to recognize the top three emotions present in an audio file using state-of-the-art machine learning techniques and models.

## Technologies Used
- **Programming Language**: Python
- **Libraries and Frameworks**:
  - Keras
  - PyTorch (Wave2Vec2)
- **Frontend**: HTML and CSS

## Features
- **Model Development**:
  - LSTM model using NAS (Neural Architecture Search) architecture.
  - LSTM teacher model with 256 and 128 nodes.
  - LSTM student model with 128 and 64 nodes.
  - GRU teacher and student models with 128 and 64 nodes.

<div style={display:flex}>
  <img src="LSTM256.png" width=600 height=500>
  <img src="GRU.png" width=600 height=300>
</div>
<hr>
  
- **Knowledge Distillation**:
  - Fine-tuning lightweight student models using knowledge distillation.
<hr>

- **Wav2Vec2 Training**:
  - Training the Wav2Vec2 model and generating predictions.
<hr>

- **Visualization**:
  - Confusion matrix.
  - Validation loss and accuracy plots for various models.

![confusionmatrix](https://github.com/user-attachments/assets/bc9cb408-342e-45f1-a7a8-1a03b881f177)
![Val](https://github.com/user-attachments/assets/30f67062-9a16-4597-92d2-d31ce1a5b48e)
<hr>

## Dataset
The **TESS Dataset** was used to train and validate the models.

![Dataset](https://github.com/user-attachments/assets/13c9d402-4d2a-4f2b-bc18-eb26755a6772)
<hr>

## Additional Features
- NAS-based architecture creation.
- Transformer-based predictions with Wave2Vec2.
- Web interface for accessing and visualizing results.

<div style={display:flex}>
  <img src="NAS.png" width=600 height=500>
  <img src="Website.png" width=600 height=500>
</div>
<hr>

## Results
- Performance metrics include confusion matrix and validation accuracy/loss plots.
- Comparative analysis of models (LSTM teacher-student, GRU teacher-student, and Wave2Vec2).

 <div style="display: flex; justify-content: center; align-items: center; height: 100vh; flex-direction: column;">
  <h5>Wav2Vec2 Model Results</h5>
  <img src="Wav2Vec2.png" width="600" height="300" />
</div>

---
