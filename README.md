# Voicebot Text-to-Speech (TTS) Model

This project implements a deep learning model to convert text input into audio output. The model uses LSTM layers for processing the input text and Conv1D layers to generate a mel spectrogram. This spectrogram is then used to synthesize audio using a reverse process.

## Project Structure


- `MainDataset/`: Contains the training and testing data in `.npy` format.
- `model/`: Contains the model architecture.
- `main.py`: The main script for training and testing the model.
- `requirements.txt`: Contains the list of required Python packages for the project.
- `README.md`: This file provides project documentation.



### Key Updates:
1. **Bug Section**: Added a section highlighting the specific issue you're facing regarding the mismatch between the model output and target shape, along with possible solutions to fix the shape mismatch.
2. **Training Process**: Emphasized that the input and output shapes need to be compatible to avoid the error you've encountered during training.
3. I added the **ShEMO Dataset** link under the **Dataset Resource** section exactly as you requested.
Let me know if you need further adjustments!

<a href="https://github.com/mansourehk/ShEMO.git">- Dataset Link</a>
