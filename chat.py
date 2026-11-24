
from src.python_chatbot import create_default_chatbot


def main() -> None:
    chatbot = create_default_chatbot()
    print("Personal Python Q&A Chatbot")
    print("Ask a Python question. Type 'topics' to list areas or 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChatbot: Goodbye!")
            break
        if question.lower() in {"exit", "quit", "bye"}:
            print("Chatbot: Goodbye!")
            break
        if question.lower() == "topics":
            print("Chatbot topics:")
            for topic in chatbot.suggest_topics(limit=40):
                print(f" - {topic}")
            continue
        result = chatbot.answer(question)
        print(f"\nChatbot: {result.answer}")
        if not result.is_fallback:
            print(f"[Topic: {result.matched_topic} | Retrieval score: {result.confidence:.3f}]")
        print()


if __name__ == "__main__":
    main()
