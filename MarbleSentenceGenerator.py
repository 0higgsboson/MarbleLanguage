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
        # Complete vocabulary (9 tokens)
        self.vocabulary = {
            'pronoun': ['I'],
            'states': ['static', 'move'],
            'directions': ['East', 'West', 'North', 'South'],
            'collision': ['bump'],
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
        """Generate a single random sentence"""
        sentence = []
        
        # Step 1: Choose sentence starter
        if self.random_event(self.probabilities['start_with_i']):
            # Start with "I"
            sentence.append('I')
            
            # Add state or movement
            if self.random_event(0.5):
                sentence.append('static')
            else:
                sentence.append('move')
                sentence.append(self.random_choice(self.vocabulary['directions']))
        else:
            # Start with "Move [Direction]"
            sentence.append('Move')
            sentence.append(self.random_choice(self.vocabulary['directions']))
        
        # Step 2: Optionally add collision
        if self.random_event(self.probabilities['include_collision']):
            sentence.append('bump')
        
        # Step 3: Optionally add continuation
        if self.random_event(self.probabilities['include_continuation']) and len(sentence) < 6:
            sentence.append('then')
            
            # Add continuation action
            if self.random_event(0.6):
                sentence.append('move')
                sentence.append(self.random_choice(self.vocabulary['directions']))
            else:
                sentence.append('static')
            
            # Optionally add another collision
            if self.random_event(0.3) and len(sentence) < 7:
                sentence.append('bump')
        
        # Step 4: Optionally end with state
        if (self.random_event(self.probabilities['end_with_state']) and 
            sentence[-1] != 'static' and 
            len(sentence) < 8):
            sentence.append('static')
        
        return ' '.join(sentence)

    def is_valid_sentence(self, sentence: str) -> bool:
        """Validate sentence grammar"""
        tokens = sentence.split(' ')
        
        # Check vocabulary compliance
        all_tokens = (
            self.vocabulary['pronoun'] +
            self.vocabulary['states'] +
            self.vocabulary['directions'] +
            self.vocabulary['collision'] +
            self.vocabulary['connector']
        )
        
        for token in tokens:
            if token not in all_tokens:
                return False
        
        # Check sentence length bounds
        if len(tokens) < 2 or len(tokens) > 8:
            return False
        
        # Check sentence must start with 'I' or 'Move'
        if tokens[0] not in ['I', 'Move']:
            return False
        
        # Basic grammar checks
        if tokens[0] == 'Move' and len(tokens) > 1:
            if tokens[1] not in self.vocabulary['directions']:
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
        filename = f"marble_sentences_{timestamp}.txt"
    
    try:
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
