
# Email Spam Classification Using NLP and Machine Learning

This project is a **Machine Learning application** developed using Natural Language Processing (NLP) and Machine Learning techniques to classify emails as **Spam** or **Not Spam (Ham)**. The application provides an interactive web interface using **Streamlit** for real-time email classification.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [How to Use](#how-to-use)
- [Project Demo](#project-demo)
- [Directory Structure](#directory-structure)
- [Future Enhancements](#future-enhancements)
- [Challenges Faced](#challenges-faced)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

This project demonstrates how NLP techniques such as **text vectorization** (using `CountVectorizer`) and **machine learning models** can be used to classify emails as spam or ham. The application is deployed locally using **Streamlit**, allowing users to input email text and classify it on the fly.

---

## Technologies Used

- **Programming Language:** Python 3.8+
- **Libraries:**
  - [`Streamlit`](https://streamlit.io/): For creating the web interface.
  - [`scikit-learn`](https://scikit-learn.org/): For text vectorization and machine learning model.
  - [`pickle`](https://docs.python.org/3/library/pickle.html): For saving and loading pre-trained models and vectorizers.
  - [`pandas`](https://pandas.pydata.org/): For data manipulation and analysis.
- **Machine Learning Model:** Trained using `CountVectorizer` and `Multinomial Naive Bayes`.

---

## Features

- **Email Classification:** Enter email text to check whether it is spam or ham.
- **Interactive Web Interface:** Built using **Streamlit** for user-friendly interaction.
- **Pre-trained Model:** Leverages a pre-trained model for fast predictions.
- **Real-time Feedback:** Displays results instantly.

---

## Setup and Installation

Follow these steps to run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/AbdulSarban/P3-Spam-Email-Classification-Using-NLP-and-Machine-Learning.git
cd P3-Spam-Email-Classification-Using-NLP-and-Machine-Learning
```

### 2. Create a Virtual Environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional)

To train the model yourself, run:

```bash
python train_model.py
```

### 5. Run the Application

Start the Streamlit app:

```bash
streamlit run spamDetector.py
```

### 6. Open the Web App

Visit the URL provided in the terminal (e.g., http://localhost:8501) to interact with the application.

---

## How to Use

1. Launch the Streamlit app.
2. Enter email text in the provided input box.
3. Click the "Classify" button to classify the email.
4. View the result displayed as "Spam" or "Ham."

---

## Project Demo

![Screenshot 2024-12-07 011839](https://github.com/user-attachments/assets/11b7c9a0-4446-4f94-832f-f8ced42a8182)


---

## Model Training Details

### Text Preprocessing

- Tokenization and removal of punctuation.
- Conversion to lowercase.
- Stopword removal.

### Vectorization

- Used `CountVectorizer` for bag-of-words representation.

### Model

- Trained using `Multinomial Naive Bayes` for high performance on text classification tasks.

### Performance Metrics

- Achieved an accuracy of ~98% on the test dataset.

---

## Directory Structure

```bash
P3-Spam-Email-Classification-Using-NLP-and-Machine-Learning/
│
├── spamDetector.py             # Main application script
├── train_model.py              # Script for training the model
├── spam.pkl                    # Pre-trained machine learning model
├── vectorizer.pkl              # Pre-trained CountVectorizer
├── requirements.txt            # List of dependencies
├── README.md                   # Project documentation
├── screenshot.png              # Screenshot of the web interface
└── spam.csv                    # Dataset used for training
```

---

## Future Enhancements

- **Deployment:** Host the app on platforms like Heroku or AWS for broader accessibility.
- **Multilingual Support:** Extend the classification to handle multiple languages.
- **Advanced Models:** Incorporate deep learning models for improved accuracy.
- **Explainability:** Add visualizations to explain model decisions.
- **User Authentication:** Implement user logins to save classification history.

---

## Challenges Faced

- **Model Overfitting:** Resolved using cross-validation and hyperparameter tuning.
- **Text Preprocessing:** Dealt with removing noisy data and handling special characters.
- **Real-time Predictions:** Ensured low latency while maintaining accuracy.

---

## Acknowledgements

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Streamlit Tutorials](https://docs.streamlit.io/)

---


