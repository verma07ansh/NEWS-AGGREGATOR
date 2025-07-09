#!/usr/bin/env python3
"""
Test script to verify that the hallucination issue in summarization is fixed
"""

from article_summarizer import get_summarizer

def test_hallucination_fix():
    """Test the summarization with the original problematic text"""
    
    # The original article text that was causing hallucination
    original_text = """
    Full Article
    Complete story with all details
    1 min read
    # See how FEMA maps reveal camps' flood risks
    The Washington Post
    Many camps in the Guadalupe river region are adjacent or are partially inside high-risk flood zones, according to maps from the Federal Emergency Management Agency.
    Article content begins
    More than a dozen summer camps dot the banks of the Guadalupe River and its tributaries, a vast network of waterways twisting through the hills of Kerr County, Texas.
    """
    
    print("🧪 Testing Hallucination Fix")
    print("=" * 50)
    
    print("\n📄 Original Text:")
    print(original_text)
    print(f"\nText length: {len(original_text)} characters")
    
    # Get summarizer
    summarizer = get_summarizer()
    
    if not summarizer.content_extraction_available:
        print("❌ Content extraction not available")
        return False
    
    print("\n🤖 Generating Summary...")
    summary = summarizer.summarize_text(original_text)
    
    if summary:
        print(f"\n✅ Summary Generated:")
        print(f"Summary: {summary}")
        print(f"Summary length: {len(summary)} characters")
        
        # Check for hallucinated content
        hallucination_indicators = [
            "94 people",
            "28 children", 
            "confirmed dead",
            "flash floods"
        ]
        
        hallucination_found = False
        for indicator in hallucination_indicators:
            if indicator.lower() in summary.lower():
                print(f"⚠️ Potential hallucination detected: '{indicator}' found in summary but not in original")
                hallucination_found = True
        
        if not hallucination_found:
            print("✅ No obvious hallucination detected!")
            
            # Check if summary contains key information from original
            key_terms = ["Guadalupe", "camps", "FEMA", "flood", "Texas"]
            terms_found = sum(1 for term in key_terms if term.lower() in summary.lower())
            
            print(f"📊 Key terms found in summary: {terms_found}/{len(key_terms)}")
            
            if terms_found >= 3:
                print("✅ Summary appears to be relevant to original content")
                return True
            else:
                print("⚠️ Summary may not capture key information")
                return False
        else:
            print("❌ Hallucination still present")
            return False
    else:
        print("❌ No summary generated")
        return False

def test_extractive_fallback():
    """Test the extractive summarization fallback"""
    
    print("\n🔄 Testing Extractive Fallback")
    print("=" * 50)
    
    # Test text
    test_text = """
    The Federal Emergency Management Agency has released new flood risk maps for the Guadalupe River region in Texas. 
    These maps show that many summer camps in the area are located in high-risk flood zones. 
    More than a dozen camps are situated along the banks of the Guadalupe River and its tributaries. 
    The river system winds through the hills of Kerr County, creating a complex network of waterways. 
    Camp operators and local officials are reviewing the new maps to assess potential risks. 
    Some camps may need to implement additional safety measures or consider relocation.
    """
    
    summarizer = get_summarizer()
    
    # Force use of extractive fallback by calling it directly
    extractive_summary = summarizer._extractive_summarization_fallback(test_text, max_sentences=2)
    
    print(f"📄 Original text length: {len(test_text)} characters")
    print(f"📝 Extractive summary: {extractive_summary}")
    print(f"📏 Summary length: {len(extractive_summary)} characters")
    
    # Check if extractive summary makes sense
    if len(extractive_summary) > 50 and "Guadalupe" in extractive_summary:
        print("✅ Extractive fallback working correctly")
        return True
    else:
        print("❌ Extractive fallback not working properly")
        return False

if __name__ == "__main__":
    print("🚀 Testing Hallucination Fixes")
    print("=" * 60)
    
    # Run tests
    hallucination_test = test_hallucination_fix()
    extractive_test = test_extractive_fallback()
    
    print(f"\n📊 Test Results:")
    print(f"   Hallucination Fix: {'✅ PASS' if hallucination_test else '❌ FAIL'}")
    print(f"   Extractive Fallback: {'✅ PASS' if extractive_test else '❌ FAIL'}")
    
    if hallucination_test and extractive_test:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ Hallucination issue appears to be fixed")
        print("✅ Extractive fallback is working")
    else:
        print(f"\n⚠️ Some tests failed - check the implementation")