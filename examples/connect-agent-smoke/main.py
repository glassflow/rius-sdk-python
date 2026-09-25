"""Tiny sample agent used to manually verify the glassflow-connect-agent skill end to end."""


def handle(query: str) -> str:
    return f"you asked: {query}"


if __name__ == "__main__":
    print(handle("what's the weather like"))
