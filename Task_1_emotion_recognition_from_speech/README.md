# 🎙️ Emotion Recognition from Speech using Deep Learning

This project detects human emotions from raw audio using deep learning and speech signal processing techniques. It extracts MFCC (Mel-Frequency Cepstral Coefficients) features from `.wav` files and trains a CNN-LSTM hybrid model for classification.

---

## 📌 Features

- 🔊 Audio-based emotion detection (Happy, Sad, Angry, etc.)
- 📈 MFCC feature extraction using `librosa`
- 🧠 Deep learning with CNN + LSTM using `TensorFlow`
- 🎯 Tested on RAVDESS dataset
- 💡 Easily extendable to other datasets like TESS and EMO-DB

---

## 🗃️ Dataset

This project uses the [RAVDESS](https://zenodo.org/record/1188976) (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

Download and extract the dataset into:

```

E:\code-alpha\Task1\Audio\_Song\_Actors\_01-24

````

You can process all `Actor_01` to `Actor_24` folders.

---

## 🛠️ Installation

### ✅ Create a virtual environment and install dependencies

```
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

### `requirements.txt`

```
librosa
numpy
pandas
scikit-learn
matplotlib
tensorflow
resampy
```

---

## 🧪 Usage

### 1. Extract MFCCs and Prepare Data

```
X, y = load_data()
```

### 2. Train the Model

```
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))
```

### 3. Evaluate Accuracy

```
model.evaluate(X_test, y_test)
```

### 4. Predict Emotion on New Audio

```
predict_emotion("path_to_new_audio.wav")
```

---

## 📊 Model Architecture

* **Input Shape**: (40, 174, 1) MFCC features
* **Layers**:

  * Conv2D → MaxPooling2D → Dropout
  * TimeDistributed Flatten
  * LSTM
  * Dense (ReLU) → Dense (Softmax)

---

## 📈 Accuracy

Achieves \~70–80% test accuracy on 8 emotion classes using RAVDESS.

---

## 🧩 File Structure

```
emotion-recognition-speech/
│
├── emotion_recognition_from_speech.ipynb        # Jupyter Notebook with training pipeline
├── requirements.txt
├── README.md
└── Audio_Song_Actors_01-24/
    └── Actor_01/
    └── Actor_02/
        ...
```

---

## 🧠 Future Work

* Add real-time emotion detection
* Use transfer learning with YAMNet or Wav2Vec2
* Extend to multilingual datasets
* Deploy as a Flask web app

---

## 👤 Author

**Shafia Manzoor**
Feel free to connect or raise issues for suggestions or questions!

---

## 📜 License

MIT License – free to use for personal and commercial projects with credit.

