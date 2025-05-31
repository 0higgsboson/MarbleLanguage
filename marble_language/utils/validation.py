#!/usr/bin/env python3
"""
Marble Language Validation Utilities
Comprehensive validation system for enhanced marble language rules
"""

from typing import Dict, List, Set, Tuple, Optional
import re
from ..core.config import MARBLE_CONFIG, MarbleConfig


class MarbleLanguageValidator:
    """Comprehensive validator for marble language sentences and rules"""
    
    def __init__(self, config: MarbleConfig = None):
        self.config = config or MARBLE_CONFIG
    
    def validate_sentence(self, sentence: str) -> Dict[str, any]:
        """Comprehensive sentence validation with detailed feedback"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'sentence': sentence,
            'tokens': sentence.split(),
            'analysis': {}
        }
        
        tokens = sentence.split()
        validation_result['analysis'] = self._analyze_sentence_structure(tokens)
        
        # Run all validation checks
        self._check_vocabulary_compliance(tokens, validation_result)
        self._check_grammar_rules(tokens, validation_result)
        self._check_marble_uniqueness(tokens, validation_result)
        self._check_collision_rules(tokens, validation_result)
        self._check_wall_validity(tokens, validation_result)
        self._check_sentence_structure(tokens, validation_result)
        
        # Set overall validity
        validation_result['is_valid'] = len(validation_result['errors']) == 0
        
        return validation_result
    
    def _analyze_sentence_structure(self, tokens: List[str]) -> Dict[str, any]:
        """Analyze the structure and components of a sentence"""
        analysis = {
            'length': len(tokens),
            'marbles': [],
            'colors': [],
            'directions': [],
            'walls': [],
            'collisions': [],
            'sequences': []
        }
        
        # Extract marbles (color + marble pairs)
        for i in range(len(tokens) - 1):
            if (tokens[i] in self.config.vocabulary['colors'] and 
                tokens[i + 1] == 'marble'):
                analysis['marbles'].append({
                    'color': tokens[i],
                    'position': i
                })
                analysis['colors'].append(tokens[i])
        
        # Extract directions
        for i, token in enumerate(tokens):
            if token in self.config.vocabulary['directions']:
                analysis['directions'].append({
                    'direction': token,
                    'position': i
                })
        
        # Extract walls
        for i, token in enumerate(tokens):
            if token in self.config.vocabulary['walls']:
                analysis['walls'].append({
                    'wall': token,
                    'position': i
                })
        
        # Extract collision patterns
        for i in range(len(tokens) - 1):
            if tokens[i] == 'bump' and i + 1 < len(tokens) and tokens[i + 1] == 'into':
                if i + 2 < len(tokens):
                    target = tokens[i + 2]
                    if i + 3 < len(tokens) and tokens[i + 3] == 'marble':
                        target = f"{target} marble"
                    analysis['collisions'].append({
                        'position': i,
                        'target': target
                    })
        
        # Extract sequences (then statements)
        for i, token in enumerate(tokens):
            if token == 'then':
                analysis['sequences'].append({
                    'position': i,
                    'type': 'continuation'
                })
        
        return analysis
    
    def _check_vocabulary_compliance(self, tokens: List[str], result: Dict):
        """Check if all tokens are in the vocabulary"""
        all_valid_tokens = self.config.get_all_tokens()
        
        for i, token in enumerate(tokens):
            if token not in all_valid_tokens:
                result['errors'].append(f"Invalid token '{token}' at position {i}")
    
    def _check_grammar_rules(self, tokens: List[str], result: Dict):
        """Check basic grammar rules"""
        # Check sentence length
        min_len = self.config.grammar_rules['min_sentence_length']
        max_len = self.config.grammar_rules['max_sentence_length']
        
        if len(tokens) < min_len:
            result['errors'].append(f"Sentence too short ({len(tokens)} < {min_len})")
        elif len(tokens) > max_len:
            result['errors'].append(f"Sentence too long ({len(tokens)} > {max_len})")
        
        # Check must start with pronoun
        if self.config.grammar_rules['must_start_with_pronoun']:
            if not tokens or tokens[0] != 'I':
                result['errors'].append("Sentence must start with 'I'")
        
        # Check color before marble
        if self.config.grammar_rules['color_before_marble']:
            for i in range(len(tokens)):
                if tokens[i] == 'marble':
                    if i == 0 or tokens[i-1] not in self.config.vocabulary['colors']:
                        result['errors'].append(f"'marble' at position {i} must be preceded by a color")
        
        # Check direction after move
        if self.config.grammar_rules['direction_after_move']:
            for i in range(len(tokens)):
                if tokens[i] == 'move':
                    if i + 1 >= len(tokens) or tokens[i+1] not in self.config.vocabulary['directions']:
                        result['errors'].append(f"'move' at position {i} must be followed by a direction")
    
    def _check_marble_uniqueness(self, tokens: List[str], result: Dict):
        """Check that each marble has a unique color"""
        if not self.config.marble_rules['unique_colors']:
            return
        
        marble_colors = []
        for i in range(len(tokens) - 1):
            if (tokens[i] in self.config.vocabulary['colors'] and 
                tokens[i + 1] == 'marble'):
                marble_colors.append(tokens[i])
        
        # Check for duplicates
        color_counts = {}
        for color in marble_colors:
            color_counts[color] = color_counts.get(color, 0) + 1
        
        for color, count in color_counts.items():
            if count > 1:
                result['errors'].append(f"Color '{color}' used {count} times (must be unique)")
    
    def _check_collision_rules(self, tokens: List[str], result: Dict):
        """Check collision-specific rules"""
        if not self.config.marble_rules['no_self_collision']:
            return
        
        # Find collision patterns and check for self-collision
        for i in range(len(tokens) - 2):
            if tokens[i] == 'bump' and tokens[i + 1] == 'into':
                target_start = i + 2
                
                # Find the marble that's doing the bumping
                bumping_marble_color = None
                for j in range(i - 1, -1, -1):
                    if (tokens[j] in self.config.vocabulary['colors'] and 
                        j + 1 < len(tokens) and tokens[j + 1] == 'marble'):
                        bumping_marble_color = tokens[j]
                        break
                
                # Check if target is the same marble
                if (target_start < len(tokens) and 
                    tokens[target_start] == bumping_marble_color and
                    target_start + 1 < len(tokens) and 
                    tokens[target_start + 1] == 'marble'):
                    result['errors'].append(f"Self-collision detected: '{bumping_marble_color}' marble cannot bump into itself")
    
    def _check_wall_validity(self, tokens: List[str], result: Dict):
        """Check that wall names are valid"""
        for i, token in enumerate(tokens):
            if token in self.config.vocabulary['walls']:
                if not self.config.is_valid_wall_name(token):
                    result['errors'].append(f"Invalid wall name '{token}' at position {i}")
    
    def _check_sentence_structure(self, tokens: List[str], result: Dict):
        """Check overall sentence structure and flow"""
        # Check that bump is followed by into
        for i, token in enumerate(tokens):
            if token == 'bump':
                if i + 1 >= len(tokens) or tokens[i + 1] != 'into':
                    result['errors'].append(f"'bump' at position {i} must be followed by 'into'")
                elif i + 2 >= len(tokens):
                    result['errors'].append(f"'bump into' at position {i} must specify a target")
        
        # Check that into is preceded by bump
        for i, token in enumerate(tokens):
            if token == 'into':
                if i == 0 or tokens[i - 1] != 'bump':
                    result['errors'].append(f"'into' at position {i} must be preceded by 'bump'")
    
    def validate_dataset(self, sentences: List[str]) -> Dict[str, any]:
        """Validate an entire dataset of sentences"""
        dataset_result = {
            'total_sentences': len(sentences),
            'valid_sentences': 0,
            'invalid_sentences': 0,
            'error_summary': {},
            'warning_summary': {},
            'sentence_results': []
        }
        
        for i, sentence in enumerate(sentences):
            sentence_result = self.validate_sentence(sentence)
            dataset_result['sentence_results'].append(sentence_result)
            
            if sentence_result['is_valid']:
                dataset_result['valid_sentences'] += 1
            else:
                dataset_result['invalid_sentences'] += 1
                
                # Collect error statistics
                for error in sentence_result['errors']:
                    error_type = error.split(':')[0] if ':' in error else error
                    dataset_result['error_summary'][error_type] = dataset_result['error_summary'].get(error_type, 0) + 1
        
        dataset_result['validity_rate'] = dataset_result['valid_sentences'] / len(sentences) if sentences else 0
        
        return dataset_result
    
    def get_validation_report(self, validation_result: Dict) -> str:
        """Generate a human-readable validation report"""
        if validation_result.get('sentence_results'):
            # Dataset validation report
            return self._generate_dataset_report(validation_result)
        else:
            # Single sentence validation report
            return self._generate_sentence_report(validation_result)
    
    def _generate_sentence_report(self, result: Dict) -> str:
        """Generate report for single sentence validation"""
        report = []
        report.append(f"Validation Report for: '{result['sentence']}'")
        report.append("=" * 60)
        
        if result['is_valid']:
            report.append("✓ VALID SENTENCE")
        else:
            report.append("✗ INVALID SENTENCE")
        
        report.append(f"\nSentence Analysis:")
        analysis = result['analysis']
        report.append(f"  Length: {analysis['length']} tokens")
        report.append(f"  Marbles: {len(analysis['marbles'])} ({[m['color'] for m in analysis['marbles']]})")
        report.append(f"  Collisions: {len(analysis['collisions'])}")
        report.append(f"  Wall interactions: {len(analysis['walls'])}")
        
        if result['errors']:
            report.append(f"\nErrors ({len(result['errors'])}):")
            for i, error in enumerate(result['errors'], 1):
                report.append(f"  {i}. {error}")
        
        if result['warnings']:
            report.append(f"\nWarnings ({len(result['warnings'])}):")
            for i, warning in enumerate(result['warnings'], 1):
                report.append(f"  {i}. {warning}")
        
        return "\n".join(report)
    
    def _generate_dataset_report(self, result: Dict) -> str:
        """Generate report for dataset validation"""
        report = []
        report.append("Dataset Validation Report")
        report.append("=" * 50)
        
        report.append(f"Total sentences: {result['total_sentences']}")
        report.append(f"Valid sentences: {result['valid_sentences']}")
        report.append(f"Invalid sentences: {result['invalid_sentences']}")
        report.append(f"Validity rate: {result['validity_rate']:.1%}")
        
        if result['error_summary']:
            report.append(f"\nError Summary:")
            for error_type, count in result['error_summary'].items():
                report.append(f"  {error_type}: {count}")
        
        return "\n".join(report)


def validate_sentences_from_file(file_path: str, config: MarbleConfig = None, datasets_dir: str = 'datasets') -> Dict:
    """Validate sentences from a dataset file"""
    validator = MarbleLanguageValidator(config)
    
    # Handle both absolute paths and filenames in datasets directory
    import os
    if not os.path.exists(file_path) and not file_path.startswith('/'):
        # Try datasets directory
        file_path = os.path.join(datasets_dir, file_path)
    
    # Parse sentences from file
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract sentences from numbered format
        numbered_pattern = r'\d+\.\s*"([^"]+)"'
        sentences = re.findall(numbered_pattern, content)
        
        if not sentences:
            # Try array format
            array_pattern = r"\['([^']+)'"
            sentences = re.findall(array_pattern, content)
        
    except FileNotFoundError:
        return {'error': f"File not found: {file_path}"}
    except Exception as e:
        return {'error': f"Error reading file: {e}"}
    
    if not sentences:
        return {'error': "No sentences found in file"}
    
    return validator.validate_dataset(sentences)


if __name__ == "__main__":
    # Example usage
    validator = MarbleLanguageValidator()
    
    # Test sentences
    test_sentences = [
        "I red marble move east",  # Valid
        "I blue marble move north bump into green marble",  # Valid
        "I yellow marble move west bump into top",  # Valid
        "I red marble move south bump into red marble",  # Invalid - self collision
        "I purple marble move east then purple marble move west",  # Invalid - duplicate color
        "marble move north",  # Invalid - no color
        "I red marble bump into something"  # Invalid - invalid target
    ]
    
    print("Individual Sentence Validation:")
    print("=" * 50)
    
    for sentence in test_sentences:
        result = validator.validate_sentence(sentence)
        print(f"\n'{sentence}'")
        print(f"Valid: {result['is_valid']}")
        if result['errors']:
            print(f"Errors: {', '.join(result['errors'])}")
    
    print("\n" + "=" * 50)
    print("Dataset Validation:")
    
    dataset_result = validator.validate_dataset(test_sentences)
    print(validator.get_validation_report(dataset_result))