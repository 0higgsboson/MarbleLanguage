#!/usr/bin/env python3
"""
Enhanced Marble Language Sentence Generator
Generates random sentences with enhanced collision rules and unique marble colors
"""

import random
import sys
from datetime import datetime
from typing import List, Set, Dict, Tuple
from .config import MARBLE_CONFIG, validate_sentence_rules


class EnhancedMarbleSentenceGenerator:
    """Enhanced sentence generator with collision rules and wall support"""
    
    def __init__(self, config=None):
        self.config = config or MARBLE_CONFIG
        self.used_colors = set()  # Track colors used in current scene
    
    def reset_scene(self):
        """Reset used colors for a new scene"""
        self.used_colors.clear()
    
    def get_available_color(self) -> str:
        """Get a random available color that hasn't been used"""
        available_colors = [color for color in self.config.vocabulary['colors'] 
                          if color not in self.used_colors]
        if not available_colors:
            self.reset_scene()  # Reset if all colors used
            available_colors = self.config.vocabulary['colors']
        
        color = random.choice(available_colors)
        self.used_colors.add(color)
        return color
    
    def get_collision_target(self, marble_color: str) -> str:
        """Get a valid collision target (not the marble itself)"""
        targets = []
        
        # Add other colored marbles
        for color in self.used_colors:
            if color != marble_color:  # Can't collide with self
                targets.append(f"{color} marble")
        
        # Add walls
        targets.extend(self.config.vocabulary['walls'])
        
        # If no other marbles, force wall collision
        if not targets:
            targets = self.config.vocabulary['walls']
        
        return random.choice(targets)
    
    def generate_enhanced_sentence(self) -> str:
        """Generate a sentence following enhanced marble language rules"""
        sentence = []
        self.reset_scene()  # Start fresh for each sentence
        
        # Start with pronoun
        sentence.append('I')
        
        # Add primary marble
        primary_color = self.get_available_color()
        sentence.extend([primary_color, 'marble'])
        
        # Add movement
        sentence.append('move')
        sentence.append(random.choice(self.config.vocabulary['directions']))
        
        # Decide if collision occurs
        if random.random() < self.config.probabilities['include_collision']:
            sentence.extend(['bump', 'into'])
            
            # Get collision target
            if random.random() < self.config.wall_rules['wall_collision_probability']:
                # Wall collision
                wall = random.choice(self.config.vocabulary['walls'])
                sentence.append(wall)
            else:
                # Marble collision - add another marble first
                secondary_color = self.get_available_color()
                collision_target = f"{secondary_color} marble"
                sentence.extend([secondary_color, 'marble'])
        
        # Optional continuation with 'then'
        if (len(sentence) < 15 and 
            random.random() < self.config.probabilities['include_continuation']):
            sentence.append('then')
            
            # Add another marble action
            tertiary_color = self.get_available_color()
            sentence.extend([tertiary_color, 'marble', 'move'])
            sentence.append(random.choice(self.config.vocabulary['directions']))
        
        # Ensure sentence doesn't exceed max length
        if len(sentence) > self.config.grammar_rules['max_sentence_length']:
            sentence = sentence[:self.config.grammar_rules['max_sentence_length']]
        
        return ' '.join(sentence)
    
    def generate_pattern_sentence(self, pattern: str) -> str:
        """Generate sentence based on specific pattern"""
        patterns = self.config.get_sentence_patterns()
        
        if pattern not in patterns:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        template = patterns[pattern]
        self.reset_scene()
        
        # Fill template variables
        replacements = {}
        if '{color}' in template:
            replacements['color'] = self.get_available_color()
        if '{color1}' in template:
            replacements['color1'] = self.get_available_color()
        if '{color2}' in template:
            replacements['color2'] = self.get_available_color()
        if '{direction}' in template:
            replacements['direction'] = random.choice(self.config.vocabulary['directions'])
        if '{direction1}' in template:
            replacements['direction1'] = random.choice(self.config.vocabulary['directions'])
        if '{direction2}' in template:
            replacements['direction2'] = random.choice(self.config.vocabulary['directions'])
        if '{wall}' in template:
            replacements['wall'] = random.choice(self.config.vocabulary['walls'])
        
        sentence = template.format(**replacements)
        return sentence
    
    def is_valid_sentence(self, sentence: str) -> bool:
        """Enhanced validation using config rules"""
        validation_results = validate_sentence_rules(sentence, self.config)
        return all(validation_results.values())
    
    def generate_sentences(self, num_sentences: int) -> List[str]:
        """Generate specified number of unique valid sentences"""
        if num_sentences <= 0:
            raise ValueError("Number of sentences must be positive")
        
        sentences: Set[str] = set()
        max_attempts = num_sentences * 50  # Increased for unique color constraint
        attempts = 0
        
        while len(sentences) < num_sentences and attempts < max_attempts:
            sentence = self.generate_enhanced_sentence()
            
            if self.is_valid_sentence(sentence):
                sentences.add(sentence)
            
            attempts += 1
        
        # Convert to list
        result = list(sentences)
        
        # Fill remaining with pattern-based generation if needed
        patterns = list(self.config.get_sentence_patterns().keys())
        while len(result) < num_sentences:
            pattern = random.choice(patterns)
            try:
                sentence = self.generate_pattern_sentence(pattern)
                if self.is_valid_sentence(sentence) and sentence not in result:
                    result.append(sentence)
            except Exception:
                continue
            
            if len(result) >= num_sentences:
                break
        
        return result[:num_sentences]
    
    def get_generation_stats(self, sentences: List[str]) -> Dict[str, any]:
        """Get statistics about generated sentences"""
        stats = {
            'total_sentences': len(sentences),
            'unique_colors_used': set(),
            'wall_collisions': 0,
            'marble_collisions': 0,
            'average_length': 0,
            'pattern_distribution': {},
        }
        
        total_length = 0
        for sentence in sentences:
            tokens = sentence.split()
            total_length += len(tokens)
            
            # Count colors
            for token in tokens:
                if token in self.config.vocabulary['colors']:
                    stats['unique_colors_used'].add(token)
            
            # Count collision types
            if 'bump into' in sentence:
                for wall in self.config.vocabulary['walls']:
                    if wall in sentence:
                        stats['wall_collisions'] += 1
                        break
                else:
                    stats['marble_collisions'] += 1
        
        stats['average_length'] = total_length / len(sentences) if sentences else 0
        stats['unique_colors_used'] = len(stats['unique_colors_used'])
        
        return stats


def save_sentences_to_file(sentences: List[str], filename: str = None) -> str:
    """Save enhanced sentences to a text file with metadata"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"datasets/enhanced_dataset-{len(sentences)}_{timestamp}.txt"
    elif not filename.startswith('datasets/'):
        # Ensure files are saved to datasets directory
        filename = f"datasets/{filename}"
    
    try:
        # Ensure datasets directory exists
        import os
        os.makedirs('datasets', exist_ok=True)
        
        generator = EnhancedMarbleSentenceGenerator()
        stats = generator.get_generation_stats(sentences)
        
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"Enhanced Marble Language Dataset - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("=" * 80 + "\n\n")
            
            # Write statistics
            file.write("Dataset Statistics:\n")
            file.write(f"  Total sentences: {stats['total_sentences']}\n")
            file.write(f"  Unique colors used: {stats['unique_colors_used']}\n")
            file.write(f"  Wall collisions: {stats['wall_collisions']}\n")
            file.write(f"  Marble collisions: {stats['marble_collisions']}\n")
            file.write(f"  Average sentence length: {stats['average_length']:.1f} tokens\n\n")
            
            # Write sentences
            file.write("Sentences:\n")
            file.write("-" * 40 + "\n")
            for i, sentence in enumerate(sentences, 1):
                file.write(f'{i}. "{sentence}"\n')
            
            file.write(f"\nTotal sentences: {len(sentences)}\n")
            file.write("\nRaw array format:\n")
            file.write(str(sentences))
        
        return filename
    except IOError as e:
        print(f"Error saving to file: {e}")
        raise


def generate_enhanced_marble_sentences(num_sentences: int = 10) -> List[str]:
    """Main function to generate enhanced marble sentences"""
    generator = EnhancedMarbleSentenceGenerator()
    return generator.generate_sentences(num_sentences)


def main():
    """Execute the enhanced generator and display results"""
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
            print("Usage: python3 enhanced_generator.py [num_sentences] [filename]")
            sys.exit(1)
    else:
        num_sentences = 10  # Default value
    
    # Check for optional filename argument
    if len(sys.argv) > 2:
        filename = sys.argv[2]
    
    print(f"Generated {num_sentences} Enhanced Marble Language Sentences:")
    print("=" * (len(f"Generated {num_sentences} Enhanced Marble Language Sentences:") + 5))
    
    sentences = generate_enhanced_marble_sentences(num_sentences)
    
    for i, sentence in enumerate(sentences, 1):
        print(f'{i}. "{sentence}"')
    
    # Show statistics
    generator = EnhancedMarbleSentenceGenerator()
    stats = generator.get_generation_stats(sentences)
    print(f"\nGeneration Statistics:")
    print(f"  Unique colors used: {stats['unique_colors_used']}")
    print(f"  Wall collisions: {stats['wall_collisions']}")
    print(f"  Marble collisions: {stats['marble_collisions']}")
    print(f"  Average length: {stats['average_length']:.1f} tokens")
    
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