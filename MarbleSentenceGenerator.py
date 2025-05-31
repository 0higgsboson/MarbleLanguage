#!/usr/bin/env python3
"""
Marble Language Random Sentence Generator
Generates random sentences in the constrained marble language and saves to file
"""

import random
import sys
from datetime import datetime
from typing import List, Set


class MarbleSentenceGenerator:
    def __init__(self):
        # Complete vocabulary with marble colors
        self.vocabulary = {
            'pronoun': ['I'],
            'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'black', 'white'],
            'object': ['marble'],
            'states': ['move'],
            'directions': ['east', 'west', 'north', 'south'],
            'collision': ['bump', 'into'],
            'connector': ['then']
        }
        
        # Probability weights for sentence construction
        self.probabilities = {
            'start_with_i': 0.6,
            'include_collision': 0.4,
            'include_continuation': 0.3,
            'end_with_state': 0.5
        }

    def random_choice(self, array: List[str]) -> str:
        """Get random element from array"""
        return random.choice(array)

    def random_event(self, probability: float) -> bool:
        """Check if random event occurs based on probability"""
        return random.random() < probability

    def generate_sentence(self) -> str:
        """Generate a single random sentence following the marble collision pattern"""
        sentence = []
        
        # Pattern: "I [color] marble move [direction] bump into [color] marble move [direction]..."
        sentence.append('I')
        sentence.append(self.random_choice(self.vocabulary['colors']))
        sentence.append('marble')
        sentence.append('move')
        sentence.append(self.random_choice(self.vocabulary['directions']))
        
        # Continue adding collision patterns until we reach close to 20 words
        while len(sentence) < 18:  # Leave room for final marble and direction
            sentence.append('bump')
            sentence.append('into')
            sentence.append(self.random_choice(self.vocabulary['colors']))
            sentence.append('marble')
            
            # Stop if we would exceed 20 words with another move/direction
            if len(sentence) >= 16:
                break
                
            sentence.append('move')
            sentence.append(self.random_choice(self.vocabulary['directions']))
        
        # Ensure we don't exceed 20 words
        if len(sentence) > 20:
            sentence = sentence[:20]
        
        return ' '.join(sentence)

    def is_valid_sentence(self, sentence: str) -> bool:
        """Validate sentence grammar for marble collision format"""
        tokens = sentence.split(' ')
        
        # Check vocabulary compliance
        all_tokens = (
            self.vocabulary['pronoun'] +
            self.vocabulary['colors'] +
            self.vocabulary['object'] +
            self.vocabulary['states'] +
            self.vocabulary['directions'] +
            self.vocabulary['collision'] +
            self.vocabulary['connector']
        )
        
        for token in tokens:
            if token not in all_tokens:
                return False
        
        # Check sentence length bounds (max 20 words)
        if len(tokens) < 5 or len(tokens) > 20:
            return False
        
        # Check sentence must start with 'I'
        if tokens[0] != 'I':
            return False
        
        # Check basic pattern: I [color] marble move [direction]
        if len(tokens) >= 5:
            if (tokens[1] not in self.vocabulary['colors'] or
                tokens[2] != 'marble' or
                tokens[3] != 'move' or
                tokens[4] not in self.vocabulary['directions']):
                return False
        
        return True

    def generate_sentences(self, num_sentences: int) -> List[str]:
        """Generate specified number of unique random sentences"""
        if num_sentences <= 0:
            raise ValueError("Number of sentences must be positive")
        
        sentences: Set[str] = set()
        max_attempts = num_sentences * 20  # Scale attempts with sentence count
        attempts = 0
        
        while len(sentences) < num_sentences and attempts < max_attempts:
            sentence = self.generate_sentence()
            
            if self.is_valid_sentence(sentence):
                sentences.add(sentence)
            
            attempts += 1
        
        # Convert Set to List and ensure we have requested number of sentences
        result = list(sentences)
        
        # If we couldn't generate enough unique sentences, fill with additional valid ones
        while len(result) < num_sentences:
            sentence = self.generate_sentence()
            if self.is_valid_sentence(sentence):
                result.append(sentence)
        
        return result[:num_sentences]


def save_sentences_to_file(sentences: List[str], filename: str = None) -> str:
    """Save sentences to a text file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"datasets/dataset-{len(sentences)}_{timestamp}.txt"
    elif not filename.startswith('datasets/'):
        # Ensure files are saved to datasets directory
        filename = f"datasets/{filename}"
    
    try:
        # Ensure datasets directory exists
        import os
        os.makedirs('datasets', exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"Marble Language Sentences - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("=" * 70 + "\n\n")
            
            for i, sentence in enumerate(sentences, 1):
                file.write(f'{i}. "{sentence}"\n')
            
            file.write(f"\nTotal sentences: {len(sentences)}\n")
            file.write("\nRaw array format:\n")
            file.write(str(sentences))
        
        return filename
    except IOError as e:
        print(f"Error saving to file: {e}")
        raise


def generate_marble_sentences(num_sentences: int = 10) -> List[str]:
    """Main function to generate and return specified number of random marble sentences"""
    generator = MarbleSentenceGenerator()
    return generator.generate_sentences(num_sentences)


def main():
    """Execute the generator and display results"""
    # Parse command line arguments
    filename = None
    if len(sys.argv) > 1:
        try:
            num_sentences = int(sys.argv[1])
            if num_sentences <= 0:
                print("Error: Number of sentences must be positive")
                sys.exit(1)
        except ValueError:
            print("Error: Please provide a valid integer for number of sentences")
            print("Usage: python3 marble_sentence_generator.py [num_sentences] [filename]")
            sys.exit(1)
    else:
        num_sentences = 10  # Default value
    
    # Check for optional filename argument
    if len(sys.argv) > 2:
        filename = sys.argv[2]
    
    print(f"Generated {num_sentences} Marble Language Sentences:")
    print("=" * (len(f"Generated {num_sentences} Marble Language Sentences:") + 5))
    
    sentences = generate_marble_sentences(num_sentences)
    for i, sentence in enumerate(sentences, 1):
        print(f'{i}. "{sentence}"')
    
    print("\nOutput as array:")
    print(sentences)
    
    # Save to file
    try:
        saved_filename = save_sentences_to_file(sentences, filename)
        print(f"\nSentences saved to: {saved_filename}")
    except IOError:
        print("Failed to save sentences to file")
        sys.exit(1)
    
    return sentences


if __name__ == "__main__":
    main()
