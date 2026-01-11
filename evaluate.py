from collections import Counter
from typing import Optional


def is_loop(response: str, n: int = 30, k: int = 20) -> bool:
    """
    Detect if a text response contains looping based on n-gram repetition.
    
    A text is considered to contain looping if it contains any n-gram at least k times.
    
    Args:
        response: The text response to check for looping.
        n: The size of the n-gram (default: 30 for reasoning models).
        k: The minimum number of times an n-gram must appear to be considered looping
           (default: 20 for reasoning models, use 10 for instruct models).
    
    Returns:
        True if the response contains looping, False otherwise.
    
    Examples:
        >>> is_loop("short text", n=30, k=20)
        False
        >>> is_loop("abc " * 1000, n=30, k=20)  # Repeated pattern
        True
    """
    # If the response is too short to have any n-gram, it cannot be looping
    if len(response) < n:
        return False
    
    # Count all n-grams in the response
    ngram_counts = Counter()
    for i in range(len(response) - n + 1):
        ngram = response[i:i + n]
        ngram_counts[ngram] += 1
        
        # Early exit: if any n-gram reaches k occurrences, return True immediately
        if ngram_counts[ngram] >= k:
            return True
    
    return False


def is_loop_for_instruct(response: str, n: int = 30, k: int = 10) -> bool:
    """
    Detect looping for instruct models with relaxed threshold (k=10).
    
    This is a convenience wrapper for is_loop with the default k value
    adjusted for instruct models which produce shorter responses.
    
    Args:
        response: The text response to check for looping.
        n: The size of the n-gram (default: 30).
        k: The minimum repetition threshold (default: 10 for instruct models).
    
    Returns:
        True if the response contains looping, False otherwise.
    """
    return is_loop(response, n=n, k=k)


if __name__ == "__main__":
    # Test cases
    # Test 1: Short text should not be looping
    assert is_loop("Hello world") == False
    print("Test 1 passed: Short text is not looping")
    
    # Test 2: Highly repetitive text should be looping
    repetitive_text = "This is a repeated sentence. " * 100
    assert is_loop(repetitive_text, n=30, k=20) == True
    print("Test 2 passed: Repetitive text is looping")
    
    # Test 3: Normal diverse text should not be looping
    diverse_text = " ".join([f"Sentence number {i} with unique content." for i in range(100)])
    assert is_loop(diverse_text, n=30, k=20) == False
    print("Test 3 passed: Diverse text is not looping")
    
    # Test 4: Test instruct model threshold (k=10)
    moderately_repetitive = "Let me think about this problem. " * 15
    assert is_loop(moderately_repetitive, n=30, k=20) == False  # Not enough for k=20
    assert is_loop_for_instruct(moderately_repetitive, n=30, k=10) == True  # Enough for k=10
    print("Test 4 passed: Instruct model threshold works correctly")
    
    print("\nAll tests passed!")
