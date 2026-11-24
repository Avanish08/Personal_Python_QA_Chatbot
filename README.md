# Personal Python Question-Answering Chatbot (From Scratch)

A notebook-first, retrieval-based chatbot that answers Python programming questions using a knowledge base written for this project. It uses Python's standard library for all chatbot logic and implements tokenization, TF-IDF weighting, cosine similarity, ranking, confidence thresholding, and fallback handling directly in our own code.

## Project rules and scope

- The 40 question-and-answer entries in `data/knowledge_base.json` are authored for this project.
- There is no downloaded dataset, pretrained language model, external AI API, or online answer service.
- The core chatbot uses only the Python standard library. `notebook` and `ipykernel` are optional tools for opening/running the `.ipynb` notebook; they are not used by the matching algorithm.
- The bot is retrieval-based. It selects the closest authored answer; it does not generate a new explanation or guarantee that every Python question is covered.
- A simple UI is not required. The project can be demonstrated entirely in Jupyter Notebook or in the terminal.

## Folder structure

```text
Personal_Python_QA_Chatbot_From_Scratch/
├── python_qa_chatbot.ipynb       # Main step-by-step notebook
├── chat.py                       # Optional terminal chat interface
├── requirements.txt              # Optional notebook environment dependencies
├── src/
│   ├── __init__.py
│   └── python_chatbot.py          # Custom NLP/retrieval implementation
├── data/
│   ├── knowledge_base.json        # 40 project-authored Python Q&As
│   └── test_questions.json        # Example questions and expected topics
├── tests/
│   └── test_chatbot.py            # Automated unit tests
├── docs/
│   └── PROJECT_REPORT.md          # Academic project report
└── outputs/
    └── demo_transcript.txt        # Reproducible example output
```

## Requirements

- Python 3.10 or newer
- No third-party package is required to run the chatbot core or unit tests.
- To use Jupyter Notebook, install the optional packages listed in `requirements.txt`.

## Setup: Jupyter Notebook

1. Extract the project folder.
2. Open a terminal inside the project folder.
3. (Optional) Create and activate a virtual environment.
4. Install the notebook tools:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Start Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open `python_qa_chatbot.ipynb` and choose **Run All**.

The core algorithm itself requires no internet connection or external service once Python and the project files are available.

## Setup: terminal mode (no Jupyter required)

From the project root, run:

```bash
python chat.py
```

Type `topics` to list supported topics or `exit` to quit.

## Run tests

The current test suite contains 14 unit tests. From the project root:

```bash
python -m unittest discover -s tests -v
```

## Verified demonstration

At package creation, all 14 unit tests passed. The included six hand-labelled Python test questions retrieved their expected topics (6/6), and the unrelated football question correctly triggered the fallback. This is a small, hand-prepared demonstration set, not a general benchmark or proof of accuracy on arbitrary questions. The notebook was executed end-to-end.

## How the chatbot works

1. **Knowledge-base loading:** Read the locally authored JSON entries and validate their required fields and unique IDs.
2. **Tokenization:** Convert text to lowercase and extract words/identifiers with a small regular expression tokenizer. A short, explicit stop-word list removes common conversational filler while preserving useful programming terms.
3. **Search document construction:** Combine the question, alternative phrasings, keywords, topic and answer. Repeating the question and keywords gives them extra influence in matching.
4. **TF-IDF:** For each term, calculate term frequency (TF) and inverse document frequency (IDF):

   `IDF(term) = log((N + 1) / (DF(term) + 1)) + 1`

   `TF-IDF(term, document) = TF(term, document) * IDF(term)`

   Each document vector is L2-normalised.
5. **Query matching:** Convert the user question into the same vocabulary space. Calculate cosine similarity against every knowledge-base entry.
6. **Ranking:** Combine cosine similarity (80%) with query-token coverage (20%) and select the highest-scoring entry.
7. **Fallback:** If the best score is below the configured threshold, return an honest message that the current knowledge base does not contain a reliable answer.

The score is a retrieval ranking signal, not a calibrated probability that an answer is correct.

## How to add a new answer

Edit `data/knowledge_base.json` and add an object with a unique `id`, `topic`, `question`, `answer`, plus optional `alternate_questions` and `keywords`. Use valid JSON (double quotes around strings and commas between fields). Restart the notebook or terminal application so the index is rebuilt.

Example shape:

```json
{
  "id": "PY041",
  "topic": "Your new topic",
  "question": "What is your new topic?",
  "alternate_questions": ["Another way to ask"],
  "keywords": ["term one", "term two"],
  "answer": "Write an explanation in your own words."
}
```

## Important limitations

- The chatbot can answer only topics represented in its knowledge base.
- TF-IDF uses word overlap; it does not deeply understand meaning, code execution, or context across multiple turns.
- The score is not a probability or a measure of factual correctness.
- Similar questions may occasionally retrieve a neighbouring topic. Review the matched topic and answer before relying on it.
- The initial knowledge base is introductory and should be expanded with explanations written and checked by the project team.

## Academic integrity note

The implementation, explanations, test data and knowledge-base content in this package are newly written for this project. Before submission, read the code, test it in your own environment, adapt the answers in your own words, and follow your institution's rules for acknowledging assistance and software tools.
