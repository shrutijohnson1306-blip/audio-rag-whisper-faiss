from rag import ask

print("🎧 Audio RAG Ready! Ask questions (type 'exit' to quit)\n")

while True:
    q = input("Ask: ")
    if q.lower() == "exit":
        break

    answer = ask(q)
    print("\n🧠", answer, "\n")
