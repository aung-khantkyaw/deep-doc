from utils.helpers import parse_quiz_questions


def test_parse_multiple_choose_with_q_prefix():
    raw = """
Q1: What is the capital of France?
A) Berlin
B) Madrid
C) Paris
D) Rome

Q2: 2 + 2 equals?
A) 3
B) 4
C) 5
D) 6
"""
    questions = parse_quiz_questions(raw, quiz_type="Multiple Choose")

    assert len(questions) == 2
    assert questions[0]["question"] == "What is the capital of France?"
    assert questions[0]["options"] == ["Berlin", "Madrid", "Paris", "Rome"]


def test_parse_multiple_choose_with_numbered_markdown_format():
    raw = """
### 1) Which layer handles retrieval?
A. Retriever
B. UI
C. Docker
D. Browser
"""
    questions = parse_quiz_questions(raw, quiz_type="Multiple Choose")

    assert len(questions) == 1
    assert questions[0]["question"] == "Which layer handles retrieval?"
    assert questions[0]["options"] == ["Retriever", "UI", "Docker", "Browser"]


def test_parse_true_false_with_labeled_questions():
    raw = """
Question 1: The Earth revolves around the Sun.
Question 2: Python is a database engine.
"""
    questions = parse_quiz_questions(raw, quiz_type="True/False")

    assert len(questions) == 2
    assert questions[0]["options"] == ["True", "False"]
    assert questions[1]["question"] == "Python is a database engine."


def test_parse_questions_from_json_payload():
    raw = """
{
  "questions": [
    {
      "question": "What does RAG stand for?",
      "options": [
        "Retrieval-Augmented Generation",
        "Random Access Graph",
        "Recursive API Gateway",
        "Resource Allocation Group"
      ]
    }
  ]
}
"""
    questions = parse_quiz_questions(raw, quiz_type="Multiple Choose")

    assert len(questions) == 1
    assert questions[0]["question"] == "What does RAG stand for?"
    assert questions[0]["options"][0] == "Retrieval-Augmented Generation"
