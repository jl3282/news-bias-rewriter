import os

def load_text_from_file(file_path):
    """
    Load text from a given file path and return as a string.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: Cleaned text content
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or contains only whitespace
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Clean and validate text
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError(f"File is empty or contains only whitespace: {file_path}")
    
    return cleaned_text


def get_text_stats(text):
    """
    Get basic statistics about the loaded text.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Dictionary containing text statistics
    """
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return {
        'character_count': len(text),
        'word_count': len(words),
        'sentence_count': sentences,
        'avg_words_per_sentence': round(len(words) / max(sentences, 1), 1)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load article text from a file for bias analysis.")
    parser.add_argument("file_path", type=str, help="Path to the input text file.")
    parser.add_argument("--stats", action="store_true", help="Show text statistics")
    parser.add_argument("--preview", type=int, default=2000, help="Number of characters to preview (default: 2000)")
    args = parser.parse_args()

    try:
        article_text = load_text_from_file(args.file_path)
        
        print("\n=== Loaded Article Text ===\n")
        print(article_text[:args.preview])
        
        if args.preview < len(article_text):
            print(f"\n... (showing first {args.preview} of {len(article_text)} characters)")
        
        if args.stats:
            stats = get_text_stats(article_text)
            print(f"\n=== Text Statistics ===")
            print(f"Characters: {stats['character_count']:,}")
            print(f"Words: {stats['word_count']:,}")
            print(f"Sentences: {stats['sentence_count']:,}")
            print(f"Avg words/sentence: {stats['avg_words_per_sentence']}")
            
    except Exception as e:
        print(f"Error: {e}")