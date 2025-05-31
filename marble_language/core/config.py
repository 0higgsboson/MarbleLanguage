#!/usr/bin/env python3
"""
Marble Language Configuration
Defines all rules, vocabulary, and constraints for the enhanced marble language system
"""

from typing import Dict, List, Set
from enum import Enum


class WallName(Enum):
    """Named walls in the marble box"""
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


class MarbleConfig:
    """Configuration class containing all marble language rules and constraints"""
    
    def __init__(self):
        # Core vocabulary with simplified marble language (marbles self-identify as colors only)
        self.vocabulary = {
            'pronoun': ['I'],
            'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'black', 'white', 'gray', 'brown', 'cyan', 
                      'lime', 'navy', 'teal', 'coral', 'gold', 'silver', 'violet', 'indigo'],  # 20 unique colors
            # Removed 'marble' - marbles self-identify as just their color
            'states': ['move', 'still'],  # Removed 'bounce' - no bouncing concept
            'directions': ['east', 'west', 'north', 'south'],  # Removed 'away' - no bouncing
            'collision': ['bump'],  # Simplified collision
            'connector': ['then'],
            'walls': ['top', 'bottom', 'left', 'right']
            # Removed physics vocabulary - simpler language
        }
        
        # Marble physics and ecosystem rules
        self.marble_rules = {
            'unique_colors': True,  # Each marble must have a unique color
            'no_self_collision': True,  # Marbles cannot bump into themselves
            'collision_awareness': True,  # Marbles know what they bumped into
            'initial_marble_count': 20,  # Start with exactly 20 marbles
            'max_marbles_per_scene': 20,  # Maximum number of marbles (expanded to 20)
            'finite_ecosystem': True,  # Once any marble dies, no new marbles can be born
            'color_change_announcement': True,  # Color changes announce "I [new_color]"
            'destroyed_marbles_no_sentences': True,  # Destroyed marbles generate no more sentences
        }
        
        # NEW: Physical properties and movement rules
        self.physics_rules = {
            'marble_diameter': 1.0,  # Marbles are 1 inch in diameter
            'movement_speed': 5.0,  # Marbles move 5 inches per iteration
            'can_be_stationary': True,  # Marbles can be still (not moving)
            'marble_collision_randomizes_direction': True,  # Random direction after marble collision
            'wall_collision_bounces_away': True,  # Bounce away from walls (not random)
            'collision_physics_enabled': True,  # Enable physics-based collision behavior
        }
        
        # Wall system rules (simplified - no bouncing)
        self.wall_rules = {
            'named_walls': [wall.value for wall in WallName],
            'random_wall_collisions': True,  # Marbles can randomly bump into walls
            'wall_collision_probability': 0.5,  # 50% of collisions are with walls
            'top_wall_changes_color': True,  # ONLY way to change color - touching top wall
            'bottom_wall_destroys_marble': True,  # Touching bottom wall destroys marble (no bouncing)
            'left_right_walls_stop_movement': True,  # Left/right walls just stop movement
        }
        
        # Collision detection and reporting
        self.collision_rules = {
            'report_collision_target': True,  # Always specify what was bumped into
            'marble_to_marble_collision': True,  # Marble can bump into another marble
            'marble_to_wall_collision': True,  # Marble can bump into walls
            'collision_consequences': ['move', 'stop'],  # What happens after collision
        }
        
        # Sentence generation probabilities
        self.probabilities = {
            'start_with_i': 0.6,
            'include_collision': 0.7,  # Increased from 0.5 - more collisions overall
            'include_wall_collision': 0.3,
            'include_continuation': 0.3,
            'end_with_state': 0.5,
            'multi_marble_scene': 0.4,  # Probability of multiple marbles in scene
            'top_wall_collision': 0.15,  # NEW: Probability of hitting top wall (color change)
            'bottom_wall_collision': 0.1,  # NEW: Probability of hitting bottom wall (destruction)
            'color_change_probability': 0.2,  # NEW: Random color change probability
        }
        
        # Grammar constraints
        self.grammar_rules = {
            'min_sentence_length': 20,  # Minimum words per sentence (increased from 3)
            'max_sentence_length': 30,  # Maximum words per sentence
            'must_start_with_pronoun': True,  # Sentences must start with 'I'
            'color_before_marble': True,  # Color must precede 'marble'
            'direction_after_move': True,  # Direction must follow 'move'
        }
        
        # Validation rules
        self.validation_rules = {
            'check_unique_colors': True,
            'check_no_self_collision': True,
            'validate_wall_names': True,
            'validate_collision_targets': True,
        }

    def get_all_tokens(self) -> List[str]:
        """Get all valid tokens in the vocabulary"""
        all_tokens = []
        for token_list in self.vocabulary.values():
            all_tokens.extend(token_list)
        return all_tokens

    def get_collision_targets(self) -> List[str]:
        """Get all valid collision targets (colored marbles + walls)"""
        targets = []
        # Colored marbles
        for color in self.vocabulary['colors']:
            targets.append(f"{color} marble")
        # Named walls
        targets.extend(self.vocabulary['walls'])
        return targets

    def is_valid_marble_identifier(self, color: str) -> bool:
        """Check if a color is valid for marble identification"""
        return color in self.vocabulary['colors']

    def is_valid_wall_name(self, wall: str) -> bool:
        """Check if a wall name is valid"""
        return wall in self.vocabulary['walls']

    def can_marble_collide_with_target(self, marble_color: str, target: str) -> bool:
        """Check if a marble can collide with a specific target"""
        # Cannot collide with itself
        if target == f"{marble_color} marble":
            return not self.marble_rules['no_self_collision']
        return True

    def get_sentence_patterns(self) -> Dict[str, str]:
        """Define valid sentence patterns for the simplified marble language"""
        return {
            'basic_movement': "I {color} move {direction}",  # Simplified: no 'marble' word
            'stationary': "I {color} still",  # Marble can be stationary
            'marble_collision': "I {color1} move {direction} bump {color2} move {random_direction}",  # Random direction after collision
            'wall_collision_left_right': "I {color} move {direction} bump {wall}",  # Left/right walls stop movement
            'complex_sequence': "I {color1} move {direction1} bump {color2} move {direction2}",
            'wall_then_marble': "I {color1} move {direction1} bump {wall} then {color2} move {direction2}",
            'top_wall_color_change': "I {color1} move {direction} bump top I {color2}",  # ONLY color change method
            'bottom_wall_destruction': "I {color} move {direction} bump bottom",  # Destruction (no further action)
            # Removed all spontaneous color changes - only top wall changes color
        }

    def get_special_tokens(self) -> Dict[str, int]:
        """Get special tokens for model training"""
        return {
            '[PAD]': 0,
            '[BOS]': 1,
            '[EOS]': 2,
            '[UNK]': 3,
        }

    def create_enhanced_vocabulary(self) -> Dict[str, int]:
        """Create complete vocabulary including all new tokens"""
        special_tokens = list(self.get_special_tokens().keys())
        all_marble_tokens = self.get_all_tokens()
        
        vocab = {}
        token_id = 0
        
        # Add special tokens first
        for token in special_tokens:
            vocab[token] = token_id
            token_id += 1
        
        # Add all marble language tokens
        for token in all_marble_tokens:
            vocab[token] = token_id
            token_id += 1
        
        return vocab

    def get_enhanced_rules_summary(self) -> str:
        """Get a human-readable summary of all rules"""
        summary = """
Marble Language Enhanced Rules:

1. MARBLE UNIQUENESS:
   - Each marble has a unique color
   - No two marbles can share the same color in a scene
   - Available colors: {colors}

2. COLLISION RULES:
   - Marbles cannot bump into themselves
   - Marbles know what they bumped into (marble or wall)
   - Collision target must be specified

3. WALL SYSTEM:
   - Four named walls: {walls}
   - Marbles can randomly bump into walls
   - Wall names are part of the vocabulary

4. COLLISION DETECTION:
   - Marble-to-marble collisions: "bump into [color] marble"
   - Marble-to-wall collisions: "bump into [wall]"
   - All collisions must specify the target

5. SENTENCE PATTERNS:
   - Must start with "I"
   - Format: "I [color] marble move [direction]"
   - Collisions: "bump into [target]"
   - Sequences: "then [action]"

6. VALIDATION:
   - Unique color enforcement
   - Self-collision prevention
   - Valid wall name checking
   - Collision target validation
        """.format(
            colors=", ".join(self.vocabulary['colors']),
            walls=", ".join(self.vocabulary['walls'])
        )
        return summary.strip()


# Global configuration instance
MARBLE_CONFIG = MarbleConfig()


def validate_sentence_rules(sentence: str, config: MarbleConfig = None) -> Dict[str, bool]:
    """Validate a sentence against all marble language rules"""
    if config is None:
        config = MARBLE_CONFIG
    
    tokens = sentence.split()
    validation_results = {
        'valid_vocabulary': True,
        'unique_colors': True,
        'no_self_collision': True,
        'valid_walls': True,
        'valid_collision_targets': True,
        'proper_grammar': True,
    }
    
    # Check vocabulary compliance
    all_valid_tokens = config.get_all_tokens()
    for token in tokens:
        if token not in all_valid_tokens:
            validation_results['valid_vocabulary'] = False
            break
    
    # Extract colors mentioned in sentence
    colors_in_sentence = []
    for i, token in enumerate(tokens):
        if token in config.vocabulary['colors'] and i + 1 < len(tokens) and tokens[i + 1] == 'marble':
            colors_in_sentence.append(token)
    
    # Check unique colors
    if len(colors_in_sentence) != len(set(colors_in_sentence)):
        validation_results['unique_colors'] = False
    
    # Check for self-collision
    if 'bump' in tokens and 'into' in tokens:
        bump_idx = tokens.index('bump')
        if bump_idx + 2 < len(tokens):
            target_color = tokens[bump_idx + 2]
            # Find the marble color that's doing the bumping
            for i, token in enumerate(tokens[:bump_idx]):
                if token in config.vocabulary['colors'] and i + 1 < len(tokens) and tokens[i + 1] == 'marble':
                    if token == target_color:
                        validation_results['no_self_collision'] = False
                    break
    
    # Check wall names
    for token in tokens:
        if token in config.vocabulary['walls']:
            if not config.is_valid_wall_name(token):
                validation_results['valid_walls'] = False
                break
    
    return validation_results


if __name__ == "__main__":
    # Print configuration summary
    config = MarbleConfig()
    print("Marble Language Configuration")
    print("=" * 50)
    print(config.get_enhanced_rules_summary())
    print("\nVocabulary Size:", len(config.create_enhanced_vocabulary()))
    print("Available Colors:", len(config.vocabulary['colors']))
    print("Wall Names:", config.vocabulary['walls'])
    print("Collision Targets:", len(config.get_collision_targets()))