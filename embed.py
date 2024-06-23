import re
import requests
import json
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from PyPDF2 import PdfReader
from nomic import AtlasDataset
from nomic import embed  # Correct import for embed
import numpy as np
import tqdm
from tqdm.auto import tqdm
import time  # Add this import
import re  # Add this import for the clean_text function
import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Nomic Atlas API configuration
#API_KEY = os.environ.get('NOMIC_API_KEY')

#if not API_KEY:
 ###   raise ValueError("NOMIC_API_KEY environment variable is not set")

# Initialize Nomic client
#nomic.login(token=API_KEY)

# Define clean_text function at the top level
# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in tqdm(reader.pages, desc="Extracting PDF pages"):
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise

def preprocess_text(text: str) -> List[Dict[str, Any]]:
    logger.info("Starting text preprocessing")
    
    # Split text into potential stanzas
    raw_stanzas = re.split(r'(\|\| \d+ \|\|)', text)
    
    processed_stanzas = []
    current_stanza = ""
    current_marker = ""
    
    for part in raw_stanzas:
        if re.match(r'\|\| \d+ \|\|', part):
            # This is a marker
            if current_stanza:
                processed_stanzas.append({
                    "text": current_stanza.strip(),
                    "marker": current_marker
                })
                current_stanza = ""
            current_marker = part.strip()
        else:
            # This is stanza text
            current_stanza += part.strip() + " "
    
    # Add the last stanza if it exists
    if current_stanza:
        processed_stanzas.append({
            "text": current_stanza.strip(),
            "marker": current_marker
        })

    logger.info(f"Preprocessing complete. Found {len(processed_stanzas)} stanzas.")
    return processed_stanzas

def create_embeddings_batch(texts: List[str], batch_size: int = 100) -> np.ndarray:
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        batch = texts[i:i+batch_size]
        try:
            output = embed.text(
                texts=batch,
                model='nomic-embed-text-v1.5',
                task_type='search_document',
            )
            all_embeddings.extend(output['embeddings'])
        except Exception as e:
            logger.error(f"Error creating embeddings for batch {i//batch_size}: {e}")
            raise
    return np.array(all_embeddings)

def generate_topic(stanza: Dict[str, str]) -> str:
    # Tokenize the text
    tokens = word_tokenize(stanza['text'].lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # Get the most common words
    common_words = Counter(tokens).most_common(3)
    
    # Join the most common words to create a topic
    topic = " ".join([word for word, _ in common_words])
    
    return topic

def generate_topics_batch(stanzas: List[Dict[str, str]]) -> List[str]:
    return [generate_topic(stanza) for stanza in tqdm(stanzas, desc="Generating topics")]

def main():
    try:
        pdf_path = os.path.join("data", "guru_granth_sahib.pdf")
        
        logger.info("Extracting text from PDF")
        guru_granth_sahib_text = extract_text_from_pdf(pdf_path)
        
        logger.info("Preprocessing text")
        stanzas = preprocess_text(guru_granth_sahib_text)
        
        logger.info(f"Total stanzas to process: {len(stanzas)}")
        
        logger.info("Generating topics")
        topics = generate_topics_batch(stanzas)
        
        # Combine stanzas with their topics
        stanzas_with_topics = [
            {**stanza, 'topic': topic}
            for stanza, topic in zip(stanzas, topics)
        ]
        
        logger.info("Creating embeddings")
        embeddings = create_embeddings_batch([s['text'] for s in stanzas_with_topics])
        
        logger.info("Creating Atlas dataset")
        dataset = AtlasDataset(
            "guru-granth-sahib",
            description="English translation of Guru Granth Sahib with keyword-based topics",
            unique_id_field="id",
            is_public=False
        )
        
        logger.info("Preparing data for upload")
        data = [
            {
                "id": f"stanza_{i}",
                "text": stanza['text'],
                "marker": stanza['marker'],
                "topic": stanza['topic']
            }
            for i, stanza in enumerate(stanzas_with_topics)
        ]
        
        logger.info("Uploading data to Atlas")
        dataset.add_data(data=data, embeddings=embeddings)
        
        logger.info("Creating index")
        map = dataset.create_index(
            indexed_field='text',
            topic_model={
                "build_topic_model": True,
                "topic_label_field": "topic"
            },
            duplicate_detection=True,
            projection=True,
            embedding_model='nomic-embed-text-v1.5'
        )
        
        logger.info(f"Dataset created and indexed successfully. Map ID: {map.id}")
        
    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")
        raise

if __name__ == "__main__":
    main()