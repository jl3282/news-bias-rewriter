#!/usr/bin/env python3
"""
Political Bias Detection Web Application
Flask app for real-time news article bias analysis
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
os.environ['MPLCONFIGDIR'] = '/tmp'  # Speed up matplotlib on deployment
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from openai import OpenAI

app = Flask(__name__)

# Configure OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rewrite_with_gpt4(text, target_bias):
    """Rewrite the article with the desired bias using GPT-4."""
    prompt_map = {
        "left": """Rewrite the following article from a progressive/liberal perspective while maintaining all factual content. 

SPECIFIC INSTRUCTIONS:
- Keep all facts, statistics, quotes, and verifiable information exactly the same
- Change framing to emphasize: social justice, environmental protection, worker rights, government accountability, systemic inequalities
- Use language that highlights: corporate responsibility, public interest, community impact, marginalized voices
- Frame policy discussions in terms of: protecting vulnerable populations, addressing inequality, environmental sustainability
- Present business/economic issues through lens of: worker welfare, consumer protection, market regulation needs
- Emphasize collective action, government solutions, and social responsibility
- Use active voice when describing progressive actions, passive voice for opposition
- Choose words with positive connotations for progressive concepts (e.g., "investment" not "spending", "protection" not "regulation")

MAINTAIN: All factual accuracy, proper journalistic tone, direct quotes, specific numbers and dates""",

        "center": """Rewrite the following article to be completely neutral and unbiased, removing all political slant.

SPECIFIC INSTRUCTIONS:
- Keep all facts, statistics, quotes, and verifiable information exactly the same  
- Use objective, balanced language that avoids loaded terms
- Present multiple perspectives equally when they exist
- Remove adjectives that carry political implications (e.g., "controversial", "landmark", "radical", "common-sense")
- Use neutral verbs (e.g., "stated" instead of "claimed" or "admitted")
- Frame issues in terms of factual impacts rather than ideological positions
- Avoid words that imply judgment or bias toward any political viewpoint
- Present cause-and-effect relationships objectively
- Use data and expert opinions to support points rather than political rhetoric
- Maintain professional journalistic tone throughout

MAINTAIN: All factual accuracy, direct quotes, specific numbers and dates, attribution to sources""",

        "right": """Rewrite the following article from a conservative perspective while maintaining all factual content.

SPECIFIC INSTRUCTIONS:
- Keep all facts, statistics, quotes, and verifiable information exactly the same
- Change framing to emphasize: individual responsibility, free market solutions, limited government, traditional values, fiscal responsibility
- Use language that highlights: economic growth, job creation, taxpayer concerns, constitutional principles, law and order
- Frame policy discussions in terms of: reducing government interference, promoting competition, protecting freedoms, fiscal prudence
- Present business/economic issues through lens of: entrepreneurship, innovation, market efficiency, economic opportunity
- Emphasize personal accountability, private sector solutions, and traditional institutions
- Use active voice when describing conservative actions, passive voice for opposition
- Choose words with positive connotations for conservative concepts (e.g., "reform" not "cuts", "opportunity" not "deregulation")

MAINTAIN: All factual accuracy, proper journalistic tone, direct quotes, specific numbers and dates"""
    }
    
    prompt = f"{prompt_map.get(target_bias, prompt_map['center'])}\n\nOriginal Article:\n{text}\n\nRewritten Article:"
    
    try:
        # Try gpt-4o-mini first (most accessible), then gpt-3.5-turbo as fallback
        models_to_try = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"]
        
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a professional news editor skilled at rewriting articles with different political perspectives while maintaining factual accuracy."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.7,
                )
                result = response.choices[0].message.content.strip()
                return result, model  # Return both result and model name
            except Exception as model_error:
                if "model_not_found" in str(model_error) and model != models_to_try[-1]:
                    continue  # Try next model
                else:
                    raise model_error
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

class BiasDetector:
    """Political bias detection model handler."""
    
    def __init__(self, model_path='models/roberta-combined-classifier'):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label_mapping = None
        self.device = torch.device('cpu')  # Use CPU for web deployment
        self.load_model()
    
    def load_model(self):
        """Load the trained model and tokenizer."""
        try:
            print(f"Loading model from {self.model_path}...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mapping
            with open(f'{self.model_path}/label_mapping.json', 'r') as f:
                mappings = json.load(f)
                self.label_mapping = mappings['id2label']
            
            print("Model loaded successfully!")
            print(f"Label mapping: {self.label_mapping}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_bias(self, text):
        """
        Predict political bias for given text.
        
        Args:
            text (str): Article text to analyze
            
        Returns:
            dict: Prediction results with probabilities and visualization
        """
        if not text or len(text.strip()) < 10:
            return {"error": "Text too short. Please provide at least 10 characters."}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
            
            # Convert to readable format
            probs_dict = {}
            for i, prob in enumerate(probabilities[0]):
                label = self.label_mapping[str(i)]
                probs_dict[label] = float(prob)
            
            predicted_label = self.label_mapping[str(predicted_class)]
            confidence = float(probabilities[0][predicted_class])
            
            # Create visualization
            chart_url = self.create_bias_chart(probs_dict, predicted_label)
            
            return {
                "predicted_bias": predicted_label,
                "confidence": confidence,
                "probabilities": probs_dict,
                "chart_url": chart_url,
                "text_length": len(text),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def create_bias_chart(self, probabilities, predicted_label):
        """Create a visualization of bias probabilities."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Chart 1: Bar chart of probabilities
        labels = list(probabilities.keys())
        values = list(probabilities.values())
        colors = {'left': '#1f77b4', 'center': '#ff7f0e', 'right': '#d62728'}
        bar_colors = [colors[label] for label in labels]
        
        bars = ax1.bar(labels, values, color=bar_colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Probability')
        ax1.set_title('Political Bias Prediction Probabilities')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Highlight predicted class
        for i, label in enumerate(labels):
            if label == predicted_label:
                bars[i].set_edgecolor('gold')
                bars[i].set_linewidth(3)
        
        # Chart 2: Political spectrum visualization
        spectrum_pos = {'left': -1, 'center': 0, 'right': 1}
        
        # Create spectrum line
        ax2.axhline(y=0, color='black', linewidth=2, alpha=0.3)
        ax2.scatter([-1, 0, 1], [0, 0, 0], s=100, c=['#1f77b4', '#ff7f0e', '#d62728'], 
                   alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add labels
        ax2.text(-1, -0.1, 'LEFT', ha='center', va='top', fontweight='bold', fontsize=12)
        ax2.text(0, -0.1, 'CENTER', ha='center', va='top', fontweight='bold', fontsize=12)
        ax2.text(1, -0.1, 'RIGHT', ha='center', va='top', fontweight='bold', fontsize=12)
        
        # Show prediction on spectrum
        pred_pos = spectrum_pos[predicted_label]
        confidence = probabilities[predicted_label]
        
        # Add prediction marker
        ax2.scatter([pred_pos], [0.2], s=200, c='gold', marker='v', 
                   edgecolors='black', linewidth=2, zorder=5)
        ax2.text(pred_pos, 0.35, f'{predicted_label.upper()}\n{confidence:.1%}', 
                ha='center', va='bottom', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='gold', alpha=0.7))
        
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-0.2, 0.5)
        ax2.set_title('Political Spectrum Position')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        plt.tight_layout()
        
        # Convert to base64 string for web display
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        plot_url = base64.b64encode(plot_data).decode()
        return f"data:image/png;base64,{plot_url}"

# Initialize the bias detector
print("Loading model from models/roberta-combined-classifier...")
try:
    detector = BiasDetector()
    print("Model loaded successfully!")
    print(f"Label mapping: {detector.label_mapping}")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    print("This might be because model files are missing or incompatible")
    # Create a dummy detector that will show errors in the UI
    detector = None

# Preload matplotlib to avoid delays
print("Preloading matplotlib...")
try:
    plt.figure()
    plt.close()
    print("Matplotlib ready!")
except Exception as e:
    print(f"Matplotlib warning: {e}")

@app.route('/')
def index():
    """Main page with input form."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_bias():
    """Analyze political bias of submitted text."""
    try:
        if detector is None:
            return jsonify({"error": "Model not loaded. Check server logs for details."}), 500
            
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Get prediction
        result = detector.predict_bias(text)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/rewrite', methods=['POST'])
def rewrite_article():
    """Rewrite article with specified political bias."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        target_bias = data.get('target_bias', 'center')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if len(text) < 10:
            return jsonify({"error": "Text too short. Please provide at least 10 characters."}), 400
        
        if target_bias not in ['left', 'center', 'right']:
            return jsonify({"error": "Invalid target bias. Must be 'left', 'center', or 'right'."}), 400
        
        # Check if OpenAI API key is configured
        if not client.api_key:
            return jsonify({"error": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."}), 500
        
        # Rewrite the text
        rewritten_text, model_used = rewrite_with_gpt4(text, target_bias)
        
        return jsonify({
            "rewritten": rewritten_text,
            "original_length": len(text),
            "rewritten_length": len(rewritten_text),
            "target_bias": target_bias,
            "model_used": model_used,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({"error": f"Rewrite error: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": detector is not None and detector.model is not None if detector else False,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/ping')
def ping():
    """Simple ping endpoint for port detection."""
    return "pong"

if __name__ == '__main__':
    print("Starting Political Bias Detection Web App...")
    
    # Get port from environment variable (for deployment) or default to 5002
    port = int(os.environ.get('PORT', 5002))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"Port: {port}")
    print(f"Debug mode: {debug_mode}")
    print(f"Model loaded: {detector is not None}")
    if detector:
        print(f"Model path: {detector.model_path}")
    
    print(f"🚀 Starting Flask server on 0.0.0.0:{port}")
    print("⏳ Server should be ready in a few seconds...")
    
    try:
        app.run(debug=debug_mode, host='0.0.0.0', port=port, threaded=True)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        raise 