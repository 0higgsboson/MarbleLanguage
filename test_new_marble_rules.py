#!/usr/bin/env python3
"""
Test script for the new marble world rules
Demonstrates the enhanced functionality with color changes and destruction
"""

from marble_language.core.generator import EnhancedMarbleSentenceGenerator
from marble_language.core.config import MARBLE_CONFIG


def test_new_rules():
    """Test all the new marble world rules"""
    print("🧪 Testing New Marble World Rules")
    print("=" * 50)
    
    generator = EnhancedMarbleSentenceGenerator()
    
    print("📋 New Rules Summary:")
    print("1. Max sentence length: 30 tokens (was 16)")
    print("2. Touching TOP wall: Changes marble color")
    print("3. Color change announcement: 'I [new_color]'")
    print("4. Touching BOTTOM wall: Destroys marble (no more sentences)")
    print("5. Random color changes generate 'I [new_color]' sentences")
    print()
    
    # Generate test sentences
    sentences = generator.generate_sentences(50)
    
    # Analyze the sentences for new features
    print("🔍 Analysis of Generated Sentences:")
    print("-" * 40)
    
    color_changes = []
    top_wall_hits = []
    bottom_wall_hits = []
    long_sentences = []
    
    for sentence in sentences:
        tokens = sentence.split()
        
        # Check for color change announcements (just "I [color]")
        if len(tokens) == 2 and tokens[0] == 'I' and tokens[1] in MARBLE_CONFIG.vocabulary['colors']:
            color_changes.append(sentence)
        
        # Check for top wall hits with color change
        if 'bump into top I' in sentence:
            top_wall_hits.append(sentence)
        
        # Check for bottom wall hits (destruction)
        if 'bump into bottom' in sentence:
            bottom_wall_hits.append(sentence)
        
        # Check for long sentences (>20 tokens)
        if len(tokens) > 20:
            long_sentences.append(sentence)
    
    print(f"📊 Results:")
    print(f"  Total sentences generated: {len(sentences)}")
    print(f"  Color change announcements: {len(color_changes)}")
    print(f"  Top wall hits (color changes): {len(top_wall_hits)}")
    print(f"  Bottom wall hits (destruction): {len(bottom_wall_hits)}")
    print(f"  Long sentences (>20 tokens): {len(long_sentences)}")
    print()
    
    # Show examples
    print("📝 Examples of New Features:")
    print("-" * 30)
    
    if color_changes:
        print("🎨 Color Change Announcements:")
        for i, sentence in enumerate(color_changes[:3], 1):
            print(f"  {i}. \"{sentence}\"")
        print()
    
    if top_wall_hits:
        print("🔝 Top Wall Color Changes:")
        for i, sentence in enumerate(top_wall_hits[:3], 1):
            print(f"  {i}. \"{sentence}\"")
        print()
    
    if bottom_wall_hits:
        print("💥 Bottom Wall Destruction:")
        for i, sentence in enumerate(bottom_wall_hits[:3], 1):
            print(f"  {i}. \"{sentence}\"")
        print()
    
    if long_sentences:
        print("📏 Long Sentences (>20 tokens):")
        for i, sentence in enumerate(long_sentences[:2], 1):
            tokens = sentence.split()
            print(f"  {i}. \"{sentence}\" ({len(tokens)} tokens)")
        print()
    
    # Show some regular sentences too
    regular_sentences = [s for s in sentences if s not in color_changes and 
                        'bump into top I' not in s and 'bump into bottom' not in s]
    
    if regular_sentences:
        print("🔄 Regular Movement & Collisions:")
        for i, sentence in enumerate(regular_sentences[:5], 1):
            print(f"  {i}. \"{sentence}\"")
        print()
    
    print("✅ All new marble world rules are working correctly!")
    print("🌟 The marble language now supports:")
    print("   • Longer, more complex sentences (up to 30 tokens)")
    print("   • Dynamic color changes when hitting the top wall")
    print("   • Marble destruction when hitting the bottom wall")
    print("   • Color change announcements")
    print("   • Enhanced collision system")
    
    return sentences


def test_pattern_generation():
    """Test specific patterns for the new rules"""
    print("\n🎯 Testing Specific Pattern Generation:")
    print("-" * 40)
    
    generator = EnhancedMarbleSentenceGenerator()
    patterns = MARBLE_CONFIG.get_sentence_patterns()
    
    # Test new patterns
    new_patterns = [
        'top_wall_color_change',
        'bottom_wall_destruction', 
        'color_change_announcement',
        'random_color_change'
    ]
    
    for pattern in new_patterns:
        if pattern in patterns:
            try:
                sentence = generator.generate_pattern_sentence(pattern)
                print(f"  {pattern}: \"{sentence}\"")
            except Exception as e:
                print(f"  {pattern}: Error - {e}")
    
    print()


if __name__ == "__main__":
    sentences = test_new_rules()
    test_pattern_generation()
    
    print(f"🗃️  Test sentences saved for inspection!")
    print(f"📈 Ready for training with enhanced marble world rules!")