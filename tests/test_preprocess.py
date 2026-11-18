"""
Basic tests for preprocessing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))

from preprocess import clean_text, full_clean


def test_clean_text():
    text = "Amazing product!!! Visit https://example.com @user #best 123"
    cleaned = clean_text(text)
    assert "https" not in cleaned
    assert "123" not in cleaned
    assert cleaned == cleaned.lower()


def test_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_full_clean():
    text = "This is a very good and excellent product!"
    result = full_clean(text)
    assert isinstance(result, str)


if __name__ == "__main__":
    test_clean_text()
    test_empty()
    test_full_clean()
    print("All tests passed!")
