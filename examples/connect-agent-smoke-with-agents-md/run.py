from glassflow import observe  # noqa: F401 - added here only if the skill patches this file during the dry run; harmless if unused before that


def handle(query: str) -> str:
    return f"you asked: {query}"


if __name__ == "__main__":
    print(handle("what's on the agenda today"))
