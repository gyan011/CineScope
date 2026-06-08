# 🎬 CineScope – AI Movie Review Sentiment Analyzer

CineScope is a premium movie-review sentiment analysis application built with **TensorFlow**, **Simple RNN**, and **Streamlit**. It analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative**, while also generating a dynamic **star rating**, **tone breakdown**, and **review analytics**.

The application features a cinematic UI inspired by film magazines and movie-review platforms.

---

## ✨ Features

### 🎭 Sentiment Analysis

* Classifies reviews as Positive or Negative
* Powered by a trained Simple RNN model
* Uses the IMDB Movie Reviews Dataset

### ⭐ Dynamic Star Rating

* Converts sentiment score into a 1–10 star rating
* Generates quality labels:

  * Masterpiece
  * Excellent
  * Very Good
  * Good
  * Average
  * Poor
  * Terrible

### 🎨 Cinematic User Interface

* Custom-built cinema-themed design
* Animated hero section
* Film-strip inspired layout
* Smooth transitions and micro-interactions

### 📊 Tone Breakdown

Detects emotional tone categories such as:

* Euphoric
* Positive
* Appreciative
* Mixed
* Critical
* Hostile

### 📈 Review Analytics

Provides:

* Word count
* Sentence count
* Average words per sentence
* Lexical richness score

### ⚡ Optimized Performance

* Cached model loading using Streamlit
* Fast prediction pipeline
* Lightweight deployment

---

## 🧠 Model Architecture

* Embedding Layer
* Simple RNN Layer
* Dense Output Layer
* Binary Classification

Dataset:

* IMDB Movie Review Dataset
* 50,000 labeled reviews

---

## 🛠️ Tech Stack

### Frontend

* Streamlit
* Custom HTML/CSS
* Streamlit Components

### Machine Learning

* TensorFlow
* Keras
* NumPy

### NLP

* IMDB Word Index
* Token Encoding
* Sequence Padding

---

## 📂 Project Structure

```bash
CineScope/
│
├── main.py
├── simple_rnn_imdb.h5
├── requirements.txt
├── README.md
│
└── assets/
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/cinescope.git
cd cinescope
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run main.py
```

---

## 📸 Screenshots

Add screenshots here after deployment:

* <img width="1892" height="732" alt="Screenshot 2026-06-08 154145" src="https://github.com/user-attachments/assets/b4e3e987-30a9-43dc-a45e-ae618ca89311" />

* Sentiment Result
* Star Rating Visualization
* Tone Analysis Panel

---

## 🔮 Future Enhancements

* Transformer-based sentiment analysis
* Review summarization
* Word cloud visualization
* Multi-language review support
* Emotion detection
* Movie recommendation engine

---

## 👨‍💻 Author

Gyan Ranjan

Passionate about AI, Deep Learning, NLP, and Full Stack Development.

---

## 📜 License

This project is licensed under the MIT License.
