# Project Report: Personal Python Question-Answering Chatbot

## 1. Abstract

This project develops a small personal chatbot that responds to questions about Python programming. It uses a local question-and-answer knowledge base written for the project and a retrieval method implemented with core Python. A user enters a natural-language question; the program tokenizes the text, represents the question and stored entries using TF-IDF vectors, compares them with cosine similarity, ranks the entries, and returns the answer associated with the best match. A confidence threshold prevents the system from presenting a weak match as a reliable answer.

## 2. Problem statement

Python learners often need quick explanations of programming concepts such as variables, collections, loops, functions, classes, exceptions and file handling. A simple personal tool can make a curated set of explanations easier to search. The challenge is to match different phrasings of a question to a relevant stored explanation without using an external chatbot service.

## 3. Objectives

- Create a local Python question-answering program.
- Build a small knowledge base with explanations authored for this project.
- Implement text tokenization without a ready-made NLP package.
- Implement TF-IDF weighting and cosine similarity directly in Python.
- Rank candidate answers and return the strongest match.
- Handle empty, unrelated and unsupported questions safely.
- Evaluate the program with repeatable test questions and unit tests.

## 4. Scope

The system covers introductory Python topics in its current 40-entry knowledge base. It retrieves fixed answers rather than generating novel text. It does not execute submitted code, search the internet, connect to an LLM, or remember previous conversation turns. The Jupyter notebook is the primary demonstration; a terminal interface is included as an optional alternative.

## 5. Tools and technologies

- Python 3.10+
- Python standard-library modules: `json`, `math`, `re`, `collections`, `dataclasses`, `pathlib`, `unittest`
- Jupyter Notebook as an optional interactive development environment

The matching algorithm has no dependency on scikit-learn, NLTK, spaCy, TensorFlow, an external API or a pretrained model.

## 6. System methodology

### 6.1 Knowledge-base design

Each entry contains a unique ID, topic, canonical question, alternative question phrasings, keywords and an answer. The initial file contains 40 entries prepared for this project.

### 6.2 Tokenization

A regular expression extracts letter/underscore identifiers and numeric tokens. Text is lowercased. A small stop-word list removes conversational filler while retaining programming-relevant words such as `if`, `for`, `not`, `is`, `class` and `in`.

### 6.3 TF-IDF representation

For a corpus of N searchable entries, document frequency DF(t) counts how many entries contain term t. The implementation calculates:

`IDF(t) = log((N + 1) / (DF(t) + 1)) + 1`

The term frequency is the raw number of occurrences in an entry. The product TF × IDF forms a sparse vector, which is normalised by its Euclidean length.

### 6.4 Similarity and ranking

The query is transformed using the vocabulary and IDF values calculated from the knowledge base. Cosine similarity compares the query vector with each stored vector:

`cosine(q, d) = (q · d) / (||q|| × ||d||)`

Because vectors are normalised, the implementation can calculate the dot product directly. A token-coverage measure is also calculated. The final ranking signal is:

`score = 0.80 × cosine_similarity + 0.20 × token_coverage`

The entry with the highest score is selected only when its score meets the configured threshold. This score is a heuristic retrieval measure, not a probability of correctness.

### 6.5 Fallback handling

An empty question receives a prompt to enter a question. If a query has no vocabulary overlap, or the strongest candidate is below the threshold, the chatbot explains that it does not have a reliable answer in its current knowledge base.

## 7. System architecture

```text
User question
     |
     v
Input validation
     |
     v
Custom tokenizer
     |
     v
Query TF-IDF vector
     |
     v
Cosine similarity + token coverage
     |
     v
Rank all knowledge-base entries
     |
     v
Threshold check
   /       \\
 pass      fail
  |          |
  v          v
Return      Fallback message
stored
answer
```

## 8. Testing and evaluation

The package includes a test-question file and 14 automated unit tests covering tokenization, vector normalisation, cosine similarity, known-topic retrieval, fallback behaviour, topic listing and knowledge-base validation. During package verification, all 14 unit tests passed. The six hand-labelled in-scope questions retrieved their expected topics (6/6), while the unrelated football question was sent to the fallback. This small hand-prepared test set is only a demonstration, not a general benchmark. The notebook was also executed end-to-end. Run tests with:

`python -m unittest discover -s tests -v`

A successful unit test confirms that the relevant software behaviour works for the tested inputs. It does not establish that every possible natural-language question will be answered correctly. The project should be evaluated further by adding questions written by different people, recording the expected topic, checking the retrieved topic, and reviewing incorrect matches.

## 9. Advantages

- Entirely local question matching.
- Easy-to-read implementation of core text-retrieval mathematics.
- Knowledge content can be edited without model training.
- No external API, pretrained model or downloaded dataset is required.
- Unsupported questions are not deliberately answered with invented content.

## 10. Limitations

- The quality and coverage depend on the written knowledge base.
- TF-IDF is based on word overlap and does not understand deeper meaning.
- The system does not maintain conversational context.
- A high similarity score does not guarantee factual correctness.
- The current content focuses on introductory Python topics.

## 11. Future enhancements

- Add more project-authored explanations and alternative question phrasings.
- Build a hand-labelled evaluation set and calculate precision, recall and retrieval accuracy.
- Add a simple conversation history in the notebook.
- Add an optional UI only if it improves the demonstration and is permitted by the course.
- Add a feedback mechanism so users can flag an incorrect topic match.

## 12. Conclusion

The project demonstrates a complete basic information-retrieval chatbot using Python. The tokenization, TF-IDF calculations, cosine similarity, ranking and fallback logic are implemented directly in the project source. The result is a transparent educational prototype that can be understood, tested and expanded by its developers.
