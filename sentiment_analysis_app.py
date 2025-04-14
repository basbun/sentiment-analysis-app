import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Advanced Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import seaborn as sns
import time
import re
import nltk
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Add this near the top of your file, after imports
import platform
import pkg_resources

# Get installed torch version for compatibility notes
try:
    torch_version = pkg_resources.get_distribution("torch").version
    python_version = platform.python_version()
    st.sidebar.info(f"Running with: Python {python_version}, PyTorch {torch_version}")
except:
    st.sidebar.warning("PyTorch version information unavailable")

# Set NLTK_DATA environment variable to the installed location
os.environ['NLTK_DATA'] = '/Users/basbun/nltk_data'

# Download VADER lexicon explicitly
with st.spinner("Downloading required NLTK data..."):
    nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
try:
    vader_analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    st.error(f"Error initializing VADER: {str(e)}")
    st.info("Using fallback transformer model only")
    vader_analyzer = None

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        color: #10B981;
        font-weight: 600;
    }
    .sentiment-negative {
        color: #EF4444;
        font-weight: 600;
    }
    .sentiment-neutral {
        color: #6B7280;
        font-weight: 600;
    }
    .stat-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #FFFFFF;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* Add styling for metric text to be black */
    .stMetric label, .stMetric div {
        color: black !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: black !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: black !important;
    }
    /* Add styles for scrollable results container */
    .results-container {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        padding: 0.5rem;
        background-color: #F9FAFB;
        margin-bottom: 1rem;
    }
    .result-item {
        padding: 0.25rem 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 0.15rem;
        font-size: 0.9rem;
        line-height: 1.2;
    }
    .result-item:last-child {
        border-bottom: none;
        margin-bottom: 0;
    }
    .result-score {
        display: inline-block;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<div class='main-header'>Sentiment Analysis Application</div>", unsafe_allow_html=True)
st.markdown("""
This application analyzes the sentiment of your text using advanced NLP techniques. 
- Enter text in the left column (one sentence per line for individual analysis)
- View sentiment scores on the right (-1 = very negative, 1 = very positive)
- Examine statistical metrics and distribution visualization below
""")

# Add a progress bar during initial load
progress_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.01)
    progress_bar.progress(percent_complete + 1)
progress_bar.empty()

@st.cache_resource
def load_model():
    """Load pre-trained model and tokenizer with caching for performance"""
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.warning("Using VADER only for sentiment analysis")
        return None, None

# Load model and tokenizer for backup/comparison
with st.spinner("Loading sentiment analysis models..."):
    tokenizer, model = load_model()
    st.success("Models loaded successfully!")

def analyze_sentiment(text):
    """
    Analyze sentiment using VADER and convert to a score between -1 and 1
    Falls back to simpler analysis if needed
    """
    if not text.strip():
        return 0.0
    
    try:
        # Use VADER for sentiment analysis
        vader_scores = vader_analyzer.polarity_scores(text)
        sentiment_score = vader_scores['compound']
        
        # Only use transformer if it was loaded successfully
        if tokenizer is not None and model is not None and -0.05 <= sentiment_score <= 0.05 and len(text.split()) > 10:
            # Tokenize the text for the transformer model
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Get model output
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get probabilities with softmax
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to sentiment score (-1 to 1)
            negative_prob = probabilities[0, 0].item()
            positive_prob = probabilities[0, 1].item()
            
            # Only use transformer score if it's decisively non-neutral
            transformer_score = positive_prob - negative_prob
            if abs(transformer_score) > 0.3:
                sentiment_score = transformer_score
        
        return sentiment_score
    
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        # Return neutral score if everything fails
        return 0.0

def get_sentiment_label(score):
    """Convert numerical score to descriptive label with more granularity"""
    if score >= 0.8:
        return "Extremely Positive"
    elif score >= 0.6:
        return "Very Positive"
    elif score >= 0.4:
        return "Positive"
    elif score >= 0.2:
        return "Slightly Positive"
    elif score > -0.2:
        return "Neutral"
    elif score > -0.4:
        return "Slightly Negative"
    elif score > -0.7:
        return "Negative"
    elif score > -0.8:
        return "Very Negative"
    else:
        return "Extremely Negative"

def get_sentiment_color_class(score):
    """Get CSS class for sentiment coloring with more granularity"""
    if score >= 0.3:
        return "sentiment-positive"
    elif score <= -0.3:
        return "sentiment-negative"
    else:
        return "sentiment-neutral"

# Create a two-column layout
col1, col2 = st.columns(2)

# Input column
with col1:
    st.markdown("<div class='sub-header'>Input Text</div>", unsafe_allow_html=True)
    user_input = st.text_area(
        "Enter text for sentiment analysis (one sentence per line for individual analysis):",
        height=300,
        placeholder="Type or paste your text here...\n\nExample:\nI love this product, it works great!\nThe customer service was terrible.\nIt was an okay experience, nothing special."
    )
    
    # Add an "Analyze Sentiment" button for mobile users
    analyze_button = st.button("Analyze Sentiment", type="primary", use_container_width=True, 
                              help="Click to analyze the text (useful for mobile users)")

# Process text when input is provided and button is clicked or when Enter is pressed
if user_input and (analyze_button or 'last_input' not in st.session_state or st.session_state.last_input != user_input):
    # Store the current input to avoid reprocessing on reruns
    st.session_state.last_input = user_input
    
    # Add a spinner during processing
    with st.spinner("Analyzing sentiment..."):
        # Split text by lines and filter out empty lines
        texts = [line for line in user_input.strip().split('\n') if line.strip()]
        
        # Analyze sentiment for each line
        sentiment_scores = [analyze_sentiment(text) for text in texts]
        
        # Get sentiment labels
        sentiment_labels = [get_sentiment_label(score) for score in sentiment_scores]
        
        # Create a dataframe for display
        results_df = pd.DataFrame({
            'Text': texts,
            'Sentiment Score': sentiment_scores,
            'Sentiment': sentiment_labels
        })

# Output column
if user_input:
    with col2:
        st.markdown("<div class='sub-header'>Sentiment Analysis Results</div>", unsafe_allow_html=True)
        
        # Format sentiment scores with color coding in a scrollable container
        st.markdown("<div class='results-container'>", unsafe_allow_html=True)
        for i, (text, score, label) in enumerate(zip(texts, sentiment_scores, sentiment_labels)):
            # Get appropriate CSS class
            color_class = get_sentiment_color_class(score)
                
            # Show text snippet and score in a compact format
            snippet = (text[:50] + "...") if len(text) > 50 else text
            st.markdown(f"<div class='result-item'><strong>{i+1}.</strong> {snippet} <span class='result-score {color_class}'>{score:.2f} ({label})</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Statistical metrics section
    st.markdown("<div class='sub-header'>Statistical Metrics</div>", unsafe_allow_html=True)
    
    # Calculate statistics
    avg_score = np.mean(sentiment_scores)
    std_dev = np.std(sentiment_scores)
    median_score = np.median(sentiment_scores)
    min_score = min(sentiment_scores)
    max_score = max(sentiment_scores)
    
    st.markdown("<div class='stat-container'>", unsafe_allow_html=True)
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Average Score", f"{avg_score:.2f}", delta=round(avg_score, 2))
    
    with metrics_col2:
        st.metric("Standard Deviation", f"{std_dev:.2f}")
    
    with metrics_col3:
        st.metric("Median Score", f"{median_score:.2f}")
    
    metrics_col4, metrics_col5 = st.columns(2)
    
    with metrics_col4:
        st.metric("Minimum Score", f"{min_score:.2f}")
    
    with metrics_col5:
        st.metric("Maximum Score", f"{max_score:.2f}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Graphical representation
    st.markdown("<div class='sub-header'>Sentiment Distribution</div>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram with seaborn for better aesthetics
    sns.histplot(sentiment_scores, bins=20, kde=True, ax=ax, color='#3B82F6', edgecolor='white', alpha=0.7)
    ax.set_xlabel('Sentiment Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Sentiment Scores', fontsize=16, fontweight='bold')
    
    # Add grid and set x-axis limits
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xlim(-1, 1)
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='#EF4444', linestyle='--', alpha=0.7, label='Neutral (0)')
    
    # Add vertical lines for interpreting scores
    ax.axvline(x=-0.7, color='#7C3AED', linestyle=':', alpha=0.5, label='Very Negative (-0.7)')
    ax.axvline(x=-0.4, color='#8B5CF6', linestyle=':', alpha=0.5, label='Negative (-0.4)')
    ax.axvline(x=-0.2, color='#B45309', linestyle=':', alpha=0.5, label='Slightly Negative (-0.2)')
    ax.axvline(x=0.2, color='#10B981', linestyle=':', alpha=0.5, label='Slightly Positive (0.2)')
    ax.axvline(x=0.4, color='#059669', linestyle=':', alpha=0.5, label='Positive (0.4)')
    ax.axvline(x=0.6, color='#047857', linestyle=':', alpha=0.5, label='Very Positive (0.6)')
    
    # Add legend
    ax.legend()
    
    # Add average line
    ax.axvline(x=avg_score, color='black', linestyle='-', alpha=0.7, label=f'Average ({avg_score:.2f})')
    
    # Add background color for readability
    ax.set_facecolor('#F9FAFB')
    fig.patch.set_facecolor('#F9FAFB')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Additional visualization - Sentiment Gauge Chart
    st.markdown("<div class='sub-header'>Sentiment Summary Gauge</div>", unsafe_allow_html=True)
    
    # Create a gauge chart for average sentiment
    fig2 = plt.figure(figsize=(10, 2))
    ax2 = fig2.add_subplot(111)
    
    # Create a horizontal gauge from -1.0 to 1.0
    gauge_range = np.linspace(-1, 1, 100)
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(gauge_range)))
    
    # Plot the gauge background
    ax2.barh(0, 2, left=-1, height=0.5, color=colors, alpha=0.7)
    
    # Plot the average sentiment marker
    ax2.barh(0, 0.05, left=avg_score-0.025, height=0.5, color='black')
    
    # Add labels
    ax2.text(-0.9, 0, "Very Negative", va='center', fontsize=10)
    ax2.text(-0.4, 0, "Negative", va='center', fontsize=10)
    ax2.text(0, 0, "Neutral", va='center', ha='center', fontsize=10)
    ax2.text(0.4, 0, "Positive", va='center', fontsize=10)
    ax2.text(0.9, 0, "Very Positive", va='center', fontsize=10)
    
    # Add the average value text
    ax2.text(avg_score, -0.5, f"Average: {avg_score:.2f}", ha='center', fontsize=12, fontweight='bold')
    
    # Remove axis ticks and labels
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1, 1)
    ax2.set_yticks([])
    ax2.set_xticks([-1, -0.6, -0.2, 0, 0.2, 0.6, 1])
    ax2.tick_params(axis='x', labelsize=8)
    
    # Remove spines
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    st.pyplot(fig2)
    
    # Detailed Results Section
    st.markdown("<div class='sub-header'>Detailed Results</div>", unsafe_allow_html=True)
    
    # Display the dataframe with search and sort capabilities
    st.dataframe(results_df, use_container_width=True)
    
    # Add download button for results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv",
    )

else:
    # Display placeholder message when no input is provided
    with col2:
        st.markdown("<div class='sub-header'>Sentiment Score</div>", unsafe_allow_html=True)
        st.info("Enter text in the input field to see sentiment analysis results.")

# Add footer with usage instructions
st.markdown("---")
st.markdown("""
### How to Use This Tool

1. **Input**: Type or paste your text in the left column. For analysis of individual sentences, put each on a separate line.
2. **Results**: View sentiment scores in the right column on a scale from -1 (very negative) to 1 (very positive).
3. **Statistics**: Examine the average sentiment, standard deviation, and other metrics below.
4. **Visualization**: The histogram shows the distribution of sentiment scores across all your text inputs.
5. **Download**: You can download the complete results as a CSV file for further analysis.

### About Sentiment Scoring

- **-1.0 to -0.6**: Very Negative
- **-0.6 to -0.2**: Negative
- **-0.2 to 0.2**: Neutral
- **0.2 to 0.6**: Positive
- **0.6 to 1.0**: Very Positive

This application uses NLTK's VADER sentiment analysis engine, designed specifically to handle:
- Mixed sentiment expressions (e.g., "The interface is good but performance is terrible")
- Negations (e.g., "Not bad at all")
- Context-specific phrases and idioms
- Emojis and punctuation (exclamation marks can intensify sentiment)

For complex cases, it can also leverage the transformer-based deep learning model as a backup.
""")