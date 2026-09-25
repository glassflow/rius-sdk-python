def handle(query: str) -> str:
    return f"you asked: {query}"


if __name__ == "__main__":
    print(handle("what's on the agenda today"))
