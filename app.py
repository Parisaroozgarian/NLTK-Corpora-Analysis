import os
import logging
import traceback
import re
import nltk
import spacy
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

app = Flask(__name__)
CORS(app)

# Load advanced NLP models
nlp = spacy.load('en_core_web_sm')

# Ensure NLTK resources are downloaded
def download_nltk_resources():
    """Download required NLTK resources with better error handling"""
    resources = [
        'gutenberg',
        'punkt',
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4',
        'vader_lexicon'
    ]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logging.info(f"Successfully downloaded {resource}")
        except Exception as e:
            logging.error(f"Error downloading {resource}: {str(e)}")
            raise Exception(f"Failed to download required NLTK resource: {resource}")

# Custom text cleaning function
def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text

class AdvancedTextAnalyzer:
    def __init__(self, text_input, is_file=False):
        try:
            # Support both file from Gutenberg and uploaded text
            if is_file:
                with open(text_input, 'r', encoding='utf-8') as file:
                    self.raw_text = file.read()
            else:
                self.raw_text = gutenberg.raw(text_input)
            
            self.cleaned_text = clean_text(self.raw_text)
            self.words = word_tokenize(self.raw_text)
            self.sentences = sent_tokenize(self.raw_text)
            
            # SpaCy processing
            self.doc = nlp(self.raw_text)
        except Exception as e:
            logging.error(f"Text Initialization Error: {e}")
            raise

    def get_comprehensive_summary(self):
        """Advanced text summary with more insights"""
        named_entities = [(ent.text, ent.label_) for ent in self.doc.ents]
        
        return {
            'total_sentences': len(self.sentences),
            'total_words': len(self.words),
            'unique_words': len(set(self.words)),
            'avg_sentence_length': sum(len(sent.split()) for sent in self.sentences) / len(self.sentences),
            'reading_level': self._estimate_reading_level(),
            'named_entities': named_entities[:10]  # Top 10 entities
        }

    def _estimate_reading_level(self):
        """Estimate text complexity using basic metrics"""
        total_sentences = len(self.sentences)
        total_words = len(self.words)
        total_syllables = sum(self._count_syllables(word) for word in self.words)
        
        # Automated Readability Index (ARI)
        ari = 4.71 * (total_words / total_sentences) + 0.5 * (total_sentences / total_words) - 21.43
        
        # Classify reading level
        if ari < 5: return "Elementary"
        elif ari < 10: return "Middle School"
        elif ari < 15: return "High School"
        else: return "College"

    def _count_syllables(self, word):
        """Basic syllable counting"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if word.endswith("le"):
            count += 1
        if count == 0:
            count += 1
        return count

    def advanced_pos_analysis(self):
        """Part of Speech detailed analysis"""
        pos_distribution = {}
        for token in self.doc:
            pos = token.pos_
            pos_distribution[pos] = pos_distribution.get(pos, 0) + 1
        
        return sorted(pos_distribution.items(), key=lambda x: x[1], reverse=True)

    def sentiment_analysis(self):
        """Advanced sentiment analysis using multiple techniques"""
        sia = SentimentIntensityAnalyzer()
        
        # NLTK Sentiment
        nltk_sentiments = [sia.polarity_scores(sent) for sent in self.sentences]
        
        # SpaCy Sentiment (basic)
        spacy_sentiments = [token.sentiment for token in self.doc if token.has_vector]
        
        return {
            'nltk_sentiments': nltk_sentiments[:10],  # First 10 sentences
            'avg_sentiment_score': sum(s['compound'] for s in nltk_sentiments) / len(nltk_sentiments),
            'spacy_sentiment_vectors': spacy_sentiments[:10]
        }

    def generate_tfidf_analysis(self, top_n=20):
        """TF-IDF based important word extraction"""
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([self.cleaned_text])
        feature_names = vectorizer.get_feature_names_out()
        
        tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
        return sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def generate_visualizations(self):
        """Generate multiple visualization types"""
        visualizations = {}
        
        # Word Frequency Plot
        freq_dist = FreqDist(self.words)
        plt.figure(figsize=(12, 5))
        freq_dist.plot(20, cumulative=False)
        plt.title('Word Frequency Distribution')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations['word_frequency'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Part of Speech Distribution
        pos_dist = self.advanced_pos_analysis()
        plt.figure(figsize=(10, 5))
        plt.bar([pos for pos, count in pos_dist], [count for pos, count in pos_dist])
        plt.title('Part of Speech Distribution')
        plt.xlabel('Part of Speech')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations['pos_distribution'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return visualizations

# Initialization
logging.info("Initializing NLTK resources...")
download_nltk_resources()
logging.info("NLTK resources initialized successfully")

@app.route('/texts', methods=['GET'])
def list_texts():
    """List available texts"""
    return jsonify(gutenberg.fileids())

@app.route('/upload', methods=['POST'])
def upload_text():
    """Handle text file uploads"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = os.path.join('/tmp', file.filename)
        file.save(filename)
        
        # Analyze uploaded text
        analyzer = AdvancedTextAnalyzer(filename, is_file=True)
        
        return jsonify({
            'summary': analyzer.get_comprehensive_summary(),
            'pos_analysis': analyzer.advanced_pos_analysis(),
            'sentiment': analyzer.sentiment_analysis(),
            'tfidf_keywords': analyzer.generate_tfidf_analysis(),
            'visualizations': analyzer.generate_visualizations()
        })
    
    except Exception as e:
        logging.error(f"Upload Error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(filename):
            os.remove(filename)

@app.route('/analyze/<text_id>', methods=['GET'])
def analyze_text(text_id):
    """Analyze a specific text from Gutenberg corpus"""
    try:
        # Verify the text exists in Gutenberg corpus
        available_texts = gutenberg.fileids()
        if text_id not in available_texts:
            logging.error(f"Text not found: {text_id}")
            return jsonify({'error': f'Text not found: {text_id}. Available texts: {available_texts}'}), 404
        
        logging.info(f"Starting analysis for text: {text_id}")
        analyzer = AdvancedTextAnalyzer(text_id)
        
        result = {
            'summary': analyzer.get_comprehensive_summary(),
            'pos_analysis': analyzer.advanced_pos_analysis(),
            'sentiment': analyzer.sentiment_analysis(),
            'tfidf_keywords': analyzer.generate_tfidf_analysis(),
            'visualizations': analyzer.generate_visualizations()
        }
        
        logging.info(f"Analysis completed for: {text_id}")
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Analysis Error for {text_id}: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)