#!/usr/bin/env python3
"""
Debug marble sentence generation
"""

from marble_language.core.generator import EnhancedMarbleSentenceGenerator


def debug_generation():
    """Debug why only empty sentences are generated"""
    print("🔍 Debugging Sentence Generation")
    print("=" * 40)
    
    gen = EnhancedMarbleSentenceGenerator()
    
    print("1. Initial ecosystem state:")
    print(f"   Total colors available: {len(gen.config.vocabulary['colors'])}")
    print(f"   Colors: {gen.config.vocabulary['colors'][:10]}...")  # Show first 10
    print(f"   Initial marbles created: {gen.initial_marbles_created}")
    print(f"   Globally destroyed: {list(gen.globally_destroyed)}")
    print(f"   Alive count: {gen.get_alive_marble_count()}")
    
    print("\n2. After initialization:")
    gen.initialize_ecosystem()
    print(f"   Active marbles: {len(gen.active_marbles)}")
    print(f"   Active marbles dict: {dict(list(gen.active_marbles.items())[:5])}...")  # Show first 5
    print(f"   Alive count: {gen.get_alive_marble_count()}")
    
    print("\n3. Attempting sentence generation:")
    for i in range(5):
        print(f"\n   Attempt {i+1}:")
        
        # Reset scene
        gen.reset_scene()
        print(f"   - After reset: used_colors={len(gen.used_colors)}, active_marbles={len(gen.active_marbles)}")
        
        # Check alive colors
        alive_colors = [color for color in gen.active_marbles.keys() 
                       if color not in gen.destroyed_marbles and color not in gen.globally_destroyed]
        print(f"   - Alive colors available: {len(alive_colors)} -> {alive_colors[:5]}...")
        
        if not alive_colors:
            print("   - ❌ NO ALIVE COLORS FOUND!")
            break
        
        # Try to generate
        sentence = gen.generate_enhanced_sentence()
        print(f"   - Generated: \"{sentence}\"")
        
        if not sentence:
            # Debug why it's empty
            print("   - ❌ Empty sentence generated")
            print(f"   - Ecosystem finite: {gen.ecosystem_finite}")
            print(f"   - Used colors: {gen.used_colors}")
            print(f"   - Destroyed marbles: {gen.destroyed_marbles}")
            print(f"   - Globally destroyed: {gen.globally_destroyed}")


if __name__ == "__main__":
    debug_generation()