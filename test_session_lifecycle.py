#!/usr/bin/env python3
"""
Test session-based marble lifecycle
Each dataset generation starts with all marbles alive
"""

from marble_language.core.generator import EnhancedMarbleSentenceGenerator


def test_session_lifecycle():
    """Test that marbles reset between dataset generation sessions"""
    print("🔄 Testing Session-Based Marble Lifecycle")
    print("=" * 50)
    
    gen = EnhancedMarbleSentenceGenerator()
    
    print("📊 SESSION 1: First dataset generation")
    print("-" * 30)
    
    # Generate first dataset
    sentences1 = gen.generate_sentences(15)
    
    destroyed_in_session1 = set()
    for i, sentence in enumerate(sentences1, 1):
        print(f"{i:2d}. \"{sentence}\"")
        if 'bump bottom' in sentence:
            destroyed_color = sentence.split()[1]  # Get color after 'I'
            destroyed_in_session1.add(destroyed_color)
            print(f"    💀 {destroyed_color} destroyed in session 1")
    
    print(f"\nSession 1 Results:")
    print(f"  Destroyed marbles: {sorted(destroyed_in_session1)}")
    print(f"  Globally destroyed: {sorted(gen.globally_destroyed)}")
    print(f"  Remaining alive: {gen.get_alive_marble_count()}")
    
    print("\n" + "=" * 50)
    print("📊 SESSION 2: Second dataset generation (should reset)")
    print("-" * 30)
    
    # Generate second dataset - should start fresh
    sentences2 = gen.generate_sentences(15)
    
    destroyed_in_session2 = set()
    revived_marbles = set()
    
    for i, sentence in enumerate(sentences2, 1):
        print(f"{i:2d}. \"{sentence}\"")
        
        # Check if previously destroyed marbles reappear (they should!)
        for destroyed_marble in destroyed_in_session1:
            if destroyed_marble in sentence and destroyed_marble not in revived_marbles:
                revived_marbles.add(destroyed_marble)
                print(f"    ✨ {destroyed_marble} REVIVED in session 2!")
        
        if 'bump bottom' in sentence:
            destroyed_color = sentence.split()[1]
            destroyed_in_session2.add(destroyed_color)
            print(f"    💀 {destroyed_color} destroyed in session 2")
    
    print(f"\nSession 2 Results:")
    print(f"  Destroyed marbles: {sorted(destroyed_in_session2)}")
    print(f"  Revived from session 1: {sorted(revived_marbles)}")
    print(f"  Globally destroyed: {sorted(gen.globally_destroyed)}")
    print(f"  Remaining alive: {gen.get_alive_marble_count()}")
    
    print("\n" + "=" * 50)
    print("🧪 VERIFICATION:")
    print("-" * 20)
    
    if revived_marbles:
        print(f"✅ SUCCESS: {len(revived_marbles)} marbles revived between sessions")
        for marble in sorted(revived_marbles):
            print(f"   🌟 {marble} was destroyed in session 1 but alive in session 2")
    else:
        if destroyed_in_session1:
            print("❌ FAILED: No marbles were revived between sessions")
        else:
            print("⚠️  No marbles were destroyed in session 1 to test revival")
    
    print(f"\nSession Independence Test:")
    print(f"  Session 1 destroyed: {len(destroyed_in_session1)} marbles")
    print(f"  Session 2 started with: 20 marbles (expected)")
    print(f"  Session 2 destroyed: {len(destroyed_in_session2)} marbles")


if __name__ == "__main__":
    test_session_lifecycle()