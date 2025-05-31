#!/usr/bin/env python3
"""
Simple test of marble destruction rule
"""

from marble_language.core.generator import EnhancedMarbleSentenceGenerator


def test_simple_destruction():
    """Test basic marble destruction functionality"""
    print("🔬 Simple Marble Destruction Test")
    print("=" * 40)
    
    gen = EnhancedMarbleSentenceGenerator()
    
    print("1. Initial state:")
    print(f"   Alive marbles: {gen.get_alive_marble_count()}")
    print(f"   Destroyed: {list(gen.globally_destroyed)}")
    
    print("\n2. Manually destroying 'red' marble:")
    gen.globally_destroyed.add('red')
    print(f"   Destroyed: {list(gen.globally_destroyed)}")
    print(f"   Alive marbles: {gen.get_alive_marble_count()}")
    
    print("\n3. Testing sentence generation with destroyed marble:")
    
    # Try to generate sentences - 'red' should not appear
    for i in range(10):
        sentence = gen.generate_enhanced_sentence()
        print(f"   {i+1}. \"{sentence}\"")
        
        # Check if destroyed marble appears
        if 'red' in sentence:
            print(f"      ❌ ERROR: Destroyed marble 'red' appeared!")
        
        gen.reset_scene()  # Reset for next sentence
    
    print("\n4. Testing bottom wall collision simulation:")
    
    # Simulate a bottom wall collision
    print("   Simulating: 'I blue move south bump bottom'")
    
    # Manually trigger destruction logic
    gen.destroyed_marbles.add('blue')
    gen.globally_destroyed.add('blue')
    gen.ecosystem_finite = True
    
    print(f"   After destruction - Alive: {gen.get_alive_marble_count()}")
    print(f"   Destroyed: {sorted(gen.globally_destroyed)}")
    
    print("\n5. Generating sentences after blue destruction:")
    for i in range(5):
        sentence = gen.generate_enhanced_sentence()
        print(f"   {i+1}. \"{sentence}\"")
        
        if 'blue' in sentence:
            print(f"      ❌ ERROR: Destroyed marble 'blue' appeared!")
        
        gen.reset_scene()


if __name__ == "__main__":
    test_simple_destruction()