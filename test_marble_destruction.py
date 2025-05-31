#!/usr/bin/env python3
"""
Test marble destruction with permanent removal from dataset
"""

from marble_language.core.generator import EnhancedMarbleSentenceGenerator


def test_bottom_wall_destruction():
    """Test that marbles hitting bottom wall are permanently removed"""
    print("🧪 Testing Bottom Wall Destruction with Dataset Removal")
    print("=" * 60)
    
    gen = EnhancedMarbleSentenceGenerator()
    
    # Track destruction
    destroyed_marbles = set()
    
    print("\n📝 Generating sentences with destruction tracking...")
    
    for i in range(100):  # Generate up to 100 sentences
        sentence = gen.generate_enhanced_sentence()
        
        if not sentence:  # Count empty sentences too
            print(f"{i+1:2d}. (empty sentence)")
            continue
            
        print(f"{i+1:2d}. \"{sentence}\"")
        
        # Check for bottom wall destruction
        if 'bump bottom' in sentence:
            # Extract the marble color (should be after 'I')
            words = sentence.split()
            if len(words) >= 2:
                marble_color = words[1]  # 'I [color] ...'
                destroyed_marbles.add(marble_color)
                print(f"    💀 {marble_color} DESTROYED - should not appear in future sentences!")
                print(f"    📊 Destroyed so far: {sorted(destroyed_marbles)}")
                print(f"    🔢 Alive marbles remaining: {gen.get_alive_marble_count()}")
        
        # Check if destroyed marbles reappear (this should NOT happen)
        for destroyed in destroyed_marbles:
            if destroyed in sentence and 'bump bottom' not in sentence:
                print(f"    ❌ ERROR: Destroyed marble '{destroyed}' reappeared in sentence!")
        
        # Reset for next sentence (but keep destroyed marbles)
        gen.reset_scene()
        
        # Force some bottom wall collisions for testing
        if i == 10 or i == 20 or i == 30:
            # Create a test sentence with bottom wall collision
            test_colors = [c for c in gen.config.vocabulary['colors'][:5] if c not in gen.globally_destroyed]
            if test_colors:
                test_sentence = f"I {test_colors[0]} move south bump bottom"
                print(f'{i+1:2d}. "{test_sentence}" (FORCED TEST)')
                # Manually trigger destruction
                gen.globally_destroyed.add(test_colors[0])
                destroyed_marbles.add(test_colors[0])
                print(f"    💀 {test_colors[0]} DESTROYED - should not appear in future sentences!")
                print(f"    📊 Destroyed so far: {sorted(destroyed_marbles)}")
                print(f"    🔢 Alive marbles remaining: {gen.get_alive_marble_count()}")
        
        # Stop if no marbles left or we've tested enough
        if gen.get_alive_marble_count() == 0:
            print(f"\n🏁 All marbles destroyed after {i+1} sentences!")
            break
        elif i > 40:  # Stop after reasonable number of tests
            print(f"\n✋ Stopping test after {i+1} sentences to check results...")
            break
    
    print(f"\n📊 Final Results:")
    print(f"   Destroyed marbles: {sorted(destroyed_marbles)}")
    print(f"   Globally destroyed: {sorted(gen.globally_destroyed)}")
    print(f"   Remaining alive: {gen.get_alive_marble_count()}")
    
    # Verify they match
    if destroyed_marbles == gen.globally_destroyed:
        print("   ✅ Destruction tracking matches!")
    else:
        print("   ❌ Destruction tracking mismatch!")
        print(f"   Expected: {sorted(destroyed_marbles)}")
        print(f"   Actual: {sorted(gen.globally_destroyed)}")


if __name__ == "__main__":
    test_bottom_wall_destruction()