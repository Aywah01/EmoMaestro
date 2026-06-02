# 🎵 EmoMaestro

**Emotion-Based Music Generation AI**

EmoMaestro is a multimodal artificial intelligence system that analyzes a user's emotions from both facial expressions and textual diary entries, then automatically generates music that matches the detected emotional state.

The project integrates Computer Vision, Natural Language Processing (NLP), and Generative AI into a unified pipeline capable of transforming abstract emotional information into expressive musical output.

---

## 📌 Features

* Facial emotion recognition using DeepFace / custom emotion models
* Korean text sentiment analysis using KoELECTRA
* Emotion representation using the Valence-Arousal-Dominance (VAD) model
* Multimodal emotion fusion (Image + Text)
* Automatic music prompt generation
* AI music synthesis using MusicGen
* Streamlit-based graphical user interface

---

## 🏗 System Architecture

Input (Image / Text)
↓
Emotion Analysis
├── Face Emotion Model
└── Text Emotion Model
↓
VAD Mapping
↓
Multimodal Fusion
↓
Prompt Generation
↓
MusicGen
↓
Generated Music (.wav)

---

## 🧠 Technology Stack

### Computer Vision

* DeepFace
* OpenCV
* PyTorch

### Natural Language Processing

* KoELECTRA
* Hugging Face Transformers

### Music Generation

* MusicGen
* Audiocraft

### Frontend

* Streamlit

### Backend

* Python 3.10

---

## 📂 Project Structure

```text
EmoMaestro/
│
├── main.py
├── gui.py
├── pipe.py
├── prompt.py
├── music.py
├── util.py
├── vad_map.py
├── face.py
├── frame.py
├── saves/
│   └── frame_weights.pth
└── datasets/
```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd EmoMaestro
```

### 2. Create Virtual Environment

```bash
python -m venv emomaestro_env
```

Activate:

Windows:

```bash
emomaestro_env\Scripts\activate
```

Linux/macOS:

```bash
source emomaestro_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Project

Run:

```bash
python main.py
```

or

```bash
streamlit run gui.py
```

The Streamlit interface will open automatically in your browser.

---

## 📖 Usage

1. Upload a facial image (optional).
2. Enter a diary entry or emotional text.
3. Click the Generate Music button.
4. Wait for emotion analysis and music synthesis.
5. Listen to the generated music and view the detected emotional profile.

---

## 🎯 Emotion Modeling

EmoMaestro uses the Valence-Arousal-Dominance (VAD) model:

* Valence → Positive vs Negative emotion
* Arousal → Calm vs Excited state
* Dominance → Weak vs Powerful feeling

This representation enables more accurate emotional interpretation than traditional categorical emotions.

---

## 📊 Experimental Results

The system successfully demonstrated:

* Multimodal emotion fusion
* VAD-based emotional representation
* Prompt-conditioned music generation
* End-to-end automated emotion-to-music workflow

Example:

| Emotion           | Generated Music Style |
| ----------------- | --------------------- |
| Happy & Confident | Upbeat EDM            |
| Calm & Positive   | Acoustic Instrumental |
| Fearful & Tense   | Horror Soundtrack     |
| Sad & Low Energy  | Emotional Piano       |

---

## 👥 Team Members

### Team 2 – EmoMaestro

* Rujhan (Team Leader) – NLP, System Architecture, Integration
* Gabriel Osborne – Facial Emotion Recognition, Pipeline Development
* Ho Huu Nhat Minh – Backend, GUI, Audio Integration

---

## 📚 References

* MusicGen (Meta AI)
* DeepFace
* KoELECTRA
* Hugging Face Transformers
* Audiocraft

---

## 📄 License

This project was developed for the ML-2502 course project and is intended for educational and research purposes.
