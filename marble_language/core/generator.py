#!/usr/bin/env python3
"""
Enhanced Marble Language Sentence Generator
Generates random sentences with enhanced collision rules and unique marble colors
"""

import random
import sys
import json
import os
import subprocess
import webbrowser
from datetime import datetime
from typing import List, Set, Dict, Tuple
from .config import MARBLE_CONFIG, validate_sentence_rules


class EnhancedMarbleSentenceGenerator:
    """Enhanced sentence generator with collision rules and wall support"""
    
    def __init__(self, config=None):
        self.config = config or MARBLE_CONFIG
        self.used_colors = set()  # Track colors used in current scene
        self.active_marbles = {}  # Track active marbles {color: True/False}
        self.destroyed_marbles = set()  # Track marbles destroyed by bottom wall (PERMANENT removal)
        self.ecosystem_finite = False  # Track if ecosystem is now finite (any death occurred)
        self.initial_marbles_created = False  # Track if initial 20 marbles were created
        self.globally_destroyed = set()  # Track marbles destroyed globally (never appear again)
        self.first_time_speakers = set()  # Track marbles that have spoken for first time (for "born")
    
    def reset_scene(self):
        """Reset used colors for a new scene but preserve ecosystem state"""
        self.used_colors.clear()
        # DON'T clear active_marbles - these persist across scenes!
        # Only clear used_colors for the current sentence
        # Don't reset destroyed_marbles, ecosystem_finite, or globally_destroyed - these persist
        # Don't reset initial_marbles_created - this persists across scenes
    
    def get_available_color(self, exclude_destroyed: bool = True, allow_new_marbles: bool = None) -> str:
        """Get a random available color that hasn't been used"""
        if allow_new_marbles is None:
            allow_new_marbles = not self.ecosystem_finite
        
        # If ecosystem is finite and we're trying to create new marbles, return None
        if self.ecosystem_finite and allow_new_marbles and not self.used_colors:
            return None
        
        available_colors = [color for color in self.config.vocabulary['colors'] 
                          if color not in self.used_colors]
        
        # Exclude destroyed marbles if requested (both locally and globally destroyed)
        if exclude_destroyed:
            available_colors = [color for color in available_colors 
                              if color not in self.destroyed_marbles and color not in self.globally_destroyed]
        
        # In finite ecosystem, only use colors from still-alive marbles (not globally destroyed)
        if self.ecosystem_finite and not allow_new_marbles:
            alive_colors = set(self.active_marbles.keys()) - self.destroyed_marbles - self.globally_destroyed
            available_colors = [color for color in available_colors if color in alive_colors]
        
        if not available_colors:
            if exclude_destroyed and self.destroyed_marbles:
                # If only destroyed marbles left, allow using them for color changes
                available_colors = [color for color in self.config.vocabulary['colors'] 
                                  if color not in self.used_colors]
            if not available_colors:
                if not self.ecosystem_finite:
                    self.reset_scene()  # Reset if all colors used (only if ecosystem not finite)
                    available_colors = self.config.vocabulary['colors']
                else:
                    return None  # No available colors in finite ecosystem
        
        if not available_colors:
            return None
        
        color = random.choice(available_colors)
        self.used_colors.add(color)
        self.active_marbles[color] = True
        return color
    
    def get_collision_target(self, marble_color: str, current_direction: str = None, avoid_bottom_wall: bool = False) -> str:
        """Get a valid collision target (not the marble itself)"""
        targets = []
        
        # Add other colored marbles (only active ones)
        for color in self.used_colors:
            if (color != marble_color and  # Can't collide with self
                color not in self.destroyed_marbles):  # Can't collide with destroyed marbles
                targets.append(f"{color} marble")
        
        # Add walls based on physics - can only hit wall if moving toward it
        valid_walls = []
        if current_direction:
            # Physics: can only hit walls if moving toward them
            if current_direction == 'north':
                valid_walls.append('top')
            elif current_direction == 'south':
                if not avoid_bottom_wall:
                    valid_walls.append('bottom')
            elif current_direction == 'west':
                valid_walls.append('left')
            elif current_direction == 'east':
                valid_walls.append('right')
        else:
            # If stationary, can still hit any wall (bump while still)
            valid_walls = self.config.vocabulary['walls'].copy()
            if avoid_bottom_wall and 'bottom' in valid_walls:
                valid_walls.remove('bottom')
        
        # Apply weights to valid walls
        wall_weights = []
        for wall in valid_walls:
            if wall == 'top':
                weight = self.config.probabilities.get('top_wall_collision', 0.15)
            elif wall == 'bottom':
                weight = self.config.probabilities.get('bottom_wall_collision', 0.1)
            else:
                weight = 0.25  # left and right walls
            wall_weights.append((wall, weight))
        
        # Choose target type first
        if targets and random.random() > 0.5:  # 50% chance marble collision if available
            return random.choice(targets)
        elif wall_weights:
            # Weighted wall selection from valid walls
            walls, weights = zip(*wall_weights)
            return random.choices(walls, weights=weights)[0]
        else:
            # Fallback to marble collision if no valid walls
            return random.choice(targets) if targets else None
    
    # Removed generate_color_change_sentence - no spontaneous color changes
    # Color changes only happen when bumping into top wall
    
    def initialize_ecosystem(self):
        """Initialize the ecosystem with 20 marbles (excluding any globally destroyed)"""
        if not self.initial_marbles_created:
            # Create exactly 20 marbles at the start (excluding destroyed ones)
            initial_count = self.config.marble_rules.get('initial_marble_count', 20)
            all_colors = self.config.vocabulary['colors'][:initial_count]
            available_colors = [color for color in all_colors if color not in self.globally_destroyed]
            
            for color in available_colors:
                self.active_marbles[color] = True
            
            self.initial_marbles_created = True
            destroyed_count = len(self.globally_destroyed)
            if destroyed_count > 0:
                print(f"🌟 Initialized marble ecosystem with {len(available_colors)} marbles ({destroyed_count} permanently destroyed): {', '.join(available_colors)}")
            else:
                print(f"🌟 Initialized marble ecosystem with {len(available_colors)} marbles: {', '.join(available_colors)}")
    
    def get_alive_marble_count(self) -> int:
        """Get the current number of alive marbles (excluding globally destroyed)"""
        if not self.initial_marbles_created:
            initial_count = self.config.marble_rules.get('initial_marble_count', 20)
            return initial_count - len(self.globally_destroyed)
        
        # Count marbles that are still active and not globally destroyed
        initial_colors = set(self.config.vocabulary['colors'][:self.config.marble_rules.get('initial_marble_count', 20)])
        alive_colors = initial_colors - self.globally_destroyed
        return len(alive_colors)
    
    def check_resurrection(self):
        """Check if resurrection should occur (5 or fewer alive marbles)"""
        alive_count = self.get_alive_marble_count()
        if alive_count <= 5 and self.globally_destroyed:
            # Randomly choose a dead marble to resurrect
            resurrected_marble = random.choice(list(self.globally_destroyed))
            self.globally_destroyed.remove(resurrected_marble)
            self.destroyed_marbles.discard(resurrected_marble)  # Remove from destroyed set
            self.active_marbles[resurrected_marble] = True  # Mark as active
            print(f"✨ Marble {resurrected_marble} resurrected! Alive count was {alive_count}, now {alive_count + 1}")
            return resurrected_marble  # Return the resurrected marble
        return None
    
    def generate_enhanced_sentence(self) -> str:
        """Generate a sentence following enhanced marble language rules"""
        # Initialize ecosystem on first use
        if not self.initial_marbles_created:
            self.initialize_ecosystem()
        
        # If all marbles are destroyed, return empty
        alive_count = self.get_alive_marble_count()
        if alive_count == 0:
            return ""  # No more marbles, no more sentences
        
        # No spontaneous color changes - removed this section
        
        sentence = []
        
        # Check if we have any marbles left to work with
        if self.get_alive_marble_count() == 0:
            return ""
        
        # Start with pronoun
        sentence.append('I')
        
        # Add primary color (simplified - marbles self-identify as colors only)
        # Use the same logic as get_alive_marble_count for consistency
        initial_colors = set(self.config.vocabulary['colors'][:self.config.marble_rules.get('initial_marble_count', 20)])
        alive_colors = list(initial_colors - self.globally_destroyed)
        
        if not alive_colors:
            return ""  # No alive marbles left
        
        # In finite ecosystem, only use existing alive marbles
        if self.ecosystem_finite or len(self.used_colors) >= self.config.marble_rules.get('initial_marble_count', 20):
            available_colors = [color for color in alive_colors if color not in self.used_colors]
            if not available_colors and alive_colors:
                # Allow reusing colors from alive marbles if needed
                self.reset_scene()
                available_colors = alive_colors
            primary_color = random.choice(available_colors) if available_colors else alive_colors[0]
            self.used_colors.add(primary_color)
        else:
            # Still in initial phase, can use get_available_color
            primary_color = self.get_available_color(exclude_destroyed=True)
            if not primary_color:
                return ""
        sentence.append(primary_color)  # Simplified: no 'marble' word
        
        # Add "born" announcement for first-time speakers
        if primary_color not in self.first_time_speakers:
            sentence.append('born')
            self.first_time_speakers.add(primary_color)
        
        # Add movement (can be moving or still)
        current_direction = None
        if random.random() < 0.8:  # 80% chance of movement
            sentence.append('move')
            current_direction = random.choice(['east', 'west', 'north', 'south'])
            sentence.append(current_direction)
        else:  # 20% chance of being stationary
            sentence.append('still')
        
        # Decide if collision occurs
        if random.random() < self.config.probabilities['include_collision']:
            sentence.append('bump')  # Simplified: removed 'into'
            
            # Get collision target (considering physics and minimum length)
            min_length = self.config.grammar_rules['min_sentence_length']
            avoid_bottom_wall = len(sentence) + 3 < min_length  # Need at least 3 more words (bump + bottom + died)
            collision_target = self.get_collision_target(primary_color, current_direction, avoid_bottom_wall)
            
            if not collision_target:
                # No valid collision targets, skip collision
                pass
            elif collision_target in self.config.vocabulary['walls']:
                # Wall collision with physics
                sentence.append(collision_target)
                
                # Handle wall physics and special effects
                if collision_target == 'top':
                    # Top wall: color change and forced south movement
                    new_color = self.get_available_color(exclude_destroyed=False)
                    if new_color:  # Only proceed if we got a valid color
                        sentence.extend(['I', new_color])
                        # Update marble state
                        if primary_color in self.used_colors:
                            self.used_colors.remove(primary_color)
                        if primary_color in self.active_marbles:
                            del self.active_marbles[primary_color]
                        self.used_colors.add(new_color)
                        self.active_marbles[new_color] = True
                    # Physics: after hitting top wall, forced to go south
                    sentence.extend(['move', 'south'])
                    
                elif collision_target == 'bottom':
                    # Bottom wall: marble dies - but first ensure minimum length
                    min_length = self.config.grammar_rules['min_sentence_length']
                    
                    # Add padding BEFORE death to meet minimum length
                    while len(sentence) + 1 < min_length:  # +1 for 'died' word
                        # Add more actions before death to reach minimum length
                        actions_needed = min_length - len(sentence) - 1
                        if actions_needed >= 6:
                            # Add a full collision sequence (bump + wall + move + direction = 4 words)
                            padding_direction = random.choice(['east', 'west', 'north'])
                            padding_wall = {'east': 'right', 'west': 'left', 'north': 'top'}[padding_direction]
                            sentence.extend(['bump', padding_wall, 'move', random.choice(['east', 'west', 'north', 'south'])])
                        elif actions_needed >= 4:
                            # Add then + color + move + direction (4 words)
                            sentence.extend(['then', primary_color, 'move', random.choice(['east', 'west', 'north'])])
                        elif actions_needed >= 2:
                            # Add then + direction (2 words)
                            sentence.extend(['then', primary_color])
                        else:
                            # Just add single padding words
                            sentence.append('then')
                        
                        # Safety check to avoid infinite loop
                        if len(sentence) >= 25:
                            break
                    
                    # Now add death
                    sentence.append('died')
                    self.destroyed_marbles.add(primary_color)
                    self.globally_destroyed.add(primary_color)  # Never appears again in any sentence
                    if primary_color in self.active_marbles:
                        self.active_marbles[primary_color] = False
                    # Mark ecosystem as finite (no new marbles can be born)
                    self.ecosystem_finite = True
                    print(f"💀 Marble {primary_color} died by touching bottom wall - permanently removed from dataset")
                    # END SENTENCE HERE - dead marbles can't do anything else
                    marble_count = self.get_alive_marble_count()
                    return ' '.join(sentence) + f' | {marble_count}'
                    
                elif collision_target == 'left':
                    # Left wall: stops movement (no automatic bounce movement shown)
                    pass
                    
                elif collision_target == 'right':
                    # Right wall: stops movement (no automatic bounce movement shown)
                    pass
                    
            else:
                # Marble collision - random direction after collision
                # collision_target should be "color marble" but we only use color now
                target_parts = collision_target.split()
                if len(target_parts) == 2:
                    # Only use the color part, not 'marble'
                    sentence.append(target_parts[0])
                    # Add physics: random direction after marble collision
                    random_direction = random.choice(['east', 'west', 'north', 'south'])
                    sentence.extend(['move', random_direction])
        
        # Optional continuation with 'then' (but not if marble was destroyed)
        if (len(sentence) < 25 and 
            primary_color not in self.destroyed_marbles and
            random.random() < self.config.probabilities['include_continuation']):
            sentence.append('then')
            
            # Add another marble action (exclude destroyed marbles - both local and global)
            available_for_continuation = [c for c in self.used_colors 
                                        if c not in self.destroyed_marbles and c not in self.globally_destroyed]
            if available_for_continuation:
                tertiary_color = random.choice(available_for_continuation)
            else:
                tertiary_color = self.get_available_color(exclude_destroyed=True)
            
            if tertiary_color:  # Only proceed if we got a valid color
                sentence.extend([tertiary_color, 'move'])  # Simplified: no 'marble' word
                sentence.append(random.choice(self.config.vocabulary['directions']))
        
        # Ensure sentence meets minimum length requirement (but not for dead marbles)
        min_length = self.config.grammar_rules['min_sentence_length']
        while len(sentence) < min_length and primary_color not in self.destroyed_marbles:
            # Add more content to reach minimum length
            if random.random() < 0.5:
                # Add another movement
                sentence.extend(['then', primary_color, 'move', random.choice(self.config.vocabulary['directions'])])
            else:
                # Add a collision sequence
                collision_target = self.get_collision_target(primary_color, random.choice(['north', 'south', 'east', 'west']), avoid_bottom_wall=False)
                if collision_target:
                    sentence.extend(['bump', collision_target])
                    if collision_target in self.config.vocabulary['walls']:
                        # Handle wall collision properly
                        if collision_target == 'bottom':
                            # Bottom wall: marble dies - END SENTENCE IMMEDIATELY
                            sentence.append('died')
                            self.destroyed_marbles.add(primary_color)
                            self.globally_destroyed.add(primary_color)
                            if primary_color in self.active_marbles:
                                self.active_marbles[primary_color] = False
                            self.ecosystem_finite = True
                            print(f"💀 Marble {primary_color} died by touching bottom wall - permanently removed from dataset")
                            # Return immediately - don't continue padding
                            marble_count = self.get_alive_marble_count()
                            return ' '.join(sentence) + f' | {marble_count}'
                        elif collision_target == 'top':
                            # Top wall changes color but we don't add that complexity in padding
                            pass
                        elif collision_target == 'left' or collision_target == 'right':
                            # Left/right walls just stop movement
                            pass
                    else:
                        # Marble collision - add random movement
                        sentence.extend(['move', random.choice(self.config.vocabulary['directions'])])
                else:
                    # Fallback: just add movement
                    sentence.extend(['then', primary_color, 'move', random.choice(self.config.vocabulary['directions'])])
        
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
    
    def start_new_dataset_session(self):
        """Reset all marbles to alive state for a new dataset generation session"""
        print("🔄 Starting new dataset generation session - all marbles reborn!")
        self.destroyed_marbles.clear()
        self.globally_destroyed.clear()  # Reset for new session
        self.ecosystem_finite = False
        self.initial_marbles_created = False
        self.active_marbles.clear()
        self.used_colors.clear()
        self.first_time_speakers.clear()  # Reset for new session
        
    def generate_sentences(self, num_sentences: int) -> List[str]:
        """Generate specified number of sentences for a dataset (session-based destruction)"""
        if num_sentences <= 0:
            raise ValueError("Number of sentences must be positive")
        
        # Start fresh session - all marbles alive again
        self.start_new_dataset_session()
        
        sentences = []  # Use list to maintain order and track destruction
        max_attempts = num_sentences * 50
        attempts = 0
        
        while len(sentences) < num_sentences and attempts < max_attempts:
            # Reset scene for each sentence but preserve destroyed marbles (within this session)
            self.reset_scene()
            
            sentence = self.generate_enhanced_sentence()
            
            # Check for resurrection after every sentence generation
            resurrected_marble = self.check_resurrection()
            if resurrected_marble:
                # Add resurrection sentence with marble count
                alive_count = self.get_alive_marble_count()
                resurrection_sentence = f"I {resurrected_marble} born"
                resurrection_with_count = f"{resurrection_sentence} | {alive_count}"
                sentences.append(resurrection_with_count)
                print(alive_count)
            
            # Stop if all marbles are destroyed (indicated by empty sentence)
            if self.get_alive_marble_count() == 0:
                print(f"🏁 All marbles destroyed. Stopping generation at {len(sentences)} sentences.")
                break
            
            # Add sentence 
            if sentence:  # Only add non-empty sentences
                # Add marble count to the sentence for the file
                alive_count = self.get_alive_marble_count()
                sentence_with_count = f"{sentence} | {alive_count}"
                sentences.append(sentence_with_count)
                # Print just the marble count to console
                print(alive_count)
            else:
                # If sentence is empty but we have alive marbles, something is wrong
                alive_count = self.get_alive_marble_count()
                if alive_count > 0:
                    print(f"⚠️ Empty sentence generated with {alive_count} marbles alive at attempt {attempts}")
            
            attempts += 1
        
        # Generate HTML and JSON tracking files
        self.save_generation_tracking(sentences)
        
        return sentences[:num_sentences]
    
    def save_generation_tracking(self, sentences: List[str]):
        """Save HTML and JSON files for tracking marble generation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create marble_generation directory if it doesn't exist
        generation_dir = "marble_generation"
        os.makedirs(generation_dir, exist_ok=True)
        
        # Create tracking data
        tracking_data = {
            "timestamp": timestamp,
            "total_sentences": len(sentences),
            "alive_marbles": self.get_alive_marble_count(),
            "dead_marbles": len(self.globally_destroyed),
            "destroyed_marbles": sorted(list(self.globally_destroyed)),
            "ecosystem_finite": self.ecosystem_finite,
            "sentences": sentences
        }
        
        # Save JSON file
        json_filename = os.path.join(generation_dir, f"marble_generation_{timestamp}.json")
        with open(json_filename, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        print(f"📊 JSON tracking saved: {json_filename}")
        
        # Generate HTML file
        html_filename = os.path.join(generation_dir, f"marble_generation_{timestamp}.html")
        self.create_html_dashboard(tracking_data, html_filename)
        print(f"🌐 HTML dashboard saved: {html_filename}")
        
        # Try to open HTML file in browser
        self.open_in_browser(html_filename)
    
    def create_html_dashboard(self, data: Dict, filename: str):
        """Create HTML dashboard for marble generation tracking"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Marble Language Generation - {data['timestamp']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-box {{ background: #e3f2fd; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #1976d2; }}
        .stat-label {{ color: #666; font-size: 0.9em; }}
        .sentences {{ background: #fafafa; padding: 20px; border-radius: 5px; }}
        .sentence {{ margin: 5px 0; padding: 10px; background: white; border-left: 4px solid #4caf50; }}
        .sentence.death {{ border-left-color: #f44336; }}
        .sentence.birth {{ border-left-color: #2196f3; }}
        .marble-count {{ float: right; font-weight: bold; color: #666; }}
        .destroyed-list {{ background: #ffebee; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Marble Language Generation Dashboard</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{data['total_sentences']}</div>
                <div class="stat-label">Total Sentences</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{data['alive_marbles']}</div>
                <div class="stat-label">Alive Marbles</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{data['dead_marbles']}</div>
                <div class="stat-label">Dead Marbles</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{'Yes' if data['ecosystem_finite'] else 'No'}</div>
                <div class="stat-label">Ecosystem Finite</div>
            </div>
        </div>
        
        <div class="destroyed-list">
            <h3>💀 Destroyed Marbles</h3>
            <p>{', '.join(data['destroyed_marbles']) if data['destroyed_marbles'] else 'None'}</p>
        </div>
        
        <div class="sentences">
            <h3>📝 Generated Sentences</h3>
"""
        
        # Add sentences with classification
        for i, sentence in enumerate(data['sentences'], 1):
            css_class = "sentence"
            if "died" in sentence:
                css_class += " death"
            elif "born" in sentence:
                css_class += " birth"
                
            # Extract marble count from sentence
            parts = sentence.split(" | ")
            sentence_text = parts[0] if len(parts) > 1 else sentence
            marble_count = parts[1] if len(parts) > 1 else "?"
            
            html_content += f"""
            <div class="{css_class}">
                <span class="marble-count">{marble_count}</span>
                {i}. {sentence_text}
            </div>"""
        
        html_content += """
        </div>
    </div>
</body>
</html>"""
        
        with open(filename, 'w') as f:
            f.write(html_content)
    
    def open_in_browser(self, html_filename: str):
        """Try to open HTML file in browser automatically"""
        try:
            # Get absolute path
            abs_path = os.path.abspath(html_filename)
            file_url = f"file://{abs_path}"
            
            # Try different methods based on OS
            import platform
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                subprocess.run(["open", abs_path], check=False)
                print(f"🌐 Opened in browser: {abs_path}")
            elif system == "linux":
                subprocess.run(["xdg-open", abs_path], check=False)
                print(f"🌐 Opened in browser: {abs_path}")
            elif system == "windows":
                os.startfile(abs_path)
                print(f"🌐 Opened in browser: {abs_path}")
            else:
                # Fallback to webbrowser module
                webbrowser.open(file_url)
                print(f"🌐 Opened in browser: {file_url}")
                
        except Exception as e:
            print(f"⚠️  Could not auto-open browser: {e}")
            print(f"📂 Please manually open: {os.path.abspath(html_filename)}")
    
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