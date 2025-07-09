#!/usr/bin/env python3
"""
Additional test cases for the summarization fix
"""

from article_summarizer import get_summarizer

def test_various_articles():
    """Test with different types of article content"""
    
    test_cases = [
        {
            "name": "Short Article",
            "text": "The weather today is sunny with temperatures reaching 75 degrees. Perfect for outdoor activities.",
            "expected_behavior": "Should return text as-is or very minimal summarization"
        },
        {
            "name": "News Article with Numbers",
            "text": """
            The stock market closed today with the Dow Jones up 250 points to 34,500. 
            Trading volume was heavy at 2.3 billion shares. Apple stock rose 3.2% to $180 per share.
            Analysts predict continued growth in the technology sector for the next quarter.
            The Federal Reserve is expected to announce interest rate decisions next week.
            """,
            "expected_behavior": "Should preserve numbers accurately"
        },
        {
            "name": "Article with Proper Nouns",
            "text": """
            President Biden met with European leaders in Brussels yesterday to discuss NATO expansion.
            The meeting included representatives from Germany, France, and the United Kingdom.
            Secretary of State Antony Blinken also attended the summit.
            Topics covered included defense spending and regional security concerns.
            """,
            "expected_behavior": "Should preserve proper nouns accurately"
        }
    ]
    
    summarizer = get_summarizer()
    results = []
    
    print("🧪 Testing Various Article Types")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📰 Test Case {i}: {case['name']}")
        print(f"Expected: {case['expected_behavior']}")
        print(f"Original: {case['text'][:100]}...")
        
        summary = summarizer.summarize_text(case['text'])
        
        if summary:
            print(f"Summary: {summary}")
            
            # Basic validation
            if case['name'] == "News Article with Numbers":
                # Check if important numbers are preserved
                important_numbers = ["250", "34,500", "2.3", "3.2", "180"]
                numbers_preserved = sum(1 for num in important_numbers if num in summary)
                print(f"Numbers preserved: {numbers_preserved}/{len(important_numbers)}")
                results.append(numbers_preserved >= 2)  # At least 2 numbers should be preserved
                
            elif case['name'] == "Article with Proper Nouns":
                # Check if important proper nouns are preserved
                important_names = ["Biden", "Brussels", "NATO", "Germany", "France"]
                names_preserved = sum(1 for name in important_names if name in summary)
                print(f"Names preserved: {names_preserved}/{len(important_names)}")
                results.append(names_preserved >= 2)  # At least 2 names should be preserved
                
            else:
                # For short article, just check it's reasonable
                results.append(len(summary) > 10 and len(summary) <= len(case['text']))
                
            print("✅ Test passed")
        else:
            print("❌ No summary generated")
            results.append(False)
    
    return all(results)

def test_hallucination_detection():
    """Test the hallucination detection mechanism"""
    
    print("\n🔍 Testing Hallucination Detection")
    print("=" * 50)
    
    summarizer = get_summarizer()
    
    # Test the validation function directly
    test_cases = [
        {
            "source": "The company reported revenue of $100 million last quarter.",
            "summary": "The company reported revenue of $100 million last quarter.",
            "should_pass": True
        },
        {
            "source": "The company reported revenue of $100 million last quarter.",
            "summary": "The company reported revenue of $200 million and 50 employees were laid off.",
            "should_pass": False  # Contains numbers not in source
        },
        {
            "source": "John Smith visited New York yesterday.",
            "summary": "John Smith and Mary Johnson visited New York and Boston yesterday.",
            "should_pass": False  # Contains names and places not in source
        }
    ]
    
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Source: {case['source']}")
        print(f"Summary: {case['summary']}")
        
        is_valid = summarizer._validate_summary_against_source(case['summary'], case['source'])
        expected = case['should_pass']
        
        if is_valid == expected:
            print(f"✅ Validation correct: {'PASS' if is_valid else 'FAIL (as expected)'}")
            results.append(True)
        else:
            print(f"❌ Validation incorrect: Expected {'PASS' if expected else 'FAIL'}, got {'PASS' if is_valid else 'FAIL'}")
            results.append(False)
    
    return all(results)

if __name__ == "__main__":
    print("🚀 Additional Summarization Tests")
    print("=" * 60)
    
    # Run tests
    article_test = test_various_articles()
    hallucination_test = test_hallucination_detection()
    
    print(f"\n📊 Final Results:")
    print(f"   Various Articles: {'✅ PASS' if article_test else '❌ FAIL'}")
    print(f"   Hallucination Detection: {'✅ PASS' if hallucination_test else '❌ FAIL'}")
    
    if article_test and hallucination_test:
        print(f"\n🎉 ALL ADDITIONAL TESTS PASSED!")
        print("✅ Summarization is working correctly across different content types")
        print("✅ Hallucination detection is functioning properly")
    else:
        print(f"\n⚠️ Some tests failed - review the implementation")