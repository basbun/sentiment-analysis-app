# Sentiment Analysis Application

An advanced sentiment analysis web application built with Streamlit that analyzes text sentiment using NLTK's VADER and transformer-based models.

## Features

- Real-time sentiment analysis of text input
- Detailed sentiment scoring on a scale from -1 (very negative) to 1 (very positive)
- Statistical metrics (average, median, standard deviation)
- Visual representation of sentiment distribution
- Sentiment gauge visualization
- Downloadable results in CSV format

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download required NLTK data:
   ```python
   import nltk
   nltk.download('vader_lexicon')
   ```

## Usage

Run the Streamlit app:
```
streamlit run sentiment_analysis_app.py
```

The application will open in your default web browser.

## How to Use

1. Enter or paste text in the input area (left column)
   - Each line will be analyzed separately
   - For best results, enter complete sentences

2. View sentiment scores in the right column
   - Scores range from -1 (very negative) to 1 (very positive)
   - Color coding provides quick visual feedback

3. Below the main interface, you'll find:
   - Statistical metrics (average score and standard deviation)
   - A histogram showing the distribution of sentiment scores
   - A table with detailed results that can be sorted and filtered

4. The application updates in real-time as you type or modify the input text

## How It Works

The application uses two sentiment analysis models:
1. NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) - Optimized for social media text
2. DistilBERT transformer model - Used as a backup for more complex cases

For each input text, the app:
- Calculates sentiment scores
- Provides sentiment labels (Very Negative to Very Positive)
- Displays statistical metrics
- Creates visualizations of sentiment distribution
- Allows downloading results as CSV

## Technical Implementation

The application uses a pre-trained DistilBERT model from Hugging Face that has been fine-tuned for sentiment analysis. The model outputs are converted to a continuous scale between -1 and 1 for intuitive interpretation.

## License

This project is open source and available for educational and personal use.

## Last Updated

April 13, 2025

## Screenshots

[Add screenshots here]