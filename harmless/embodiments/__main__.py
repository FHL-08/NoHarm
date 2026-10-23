"""`python -m harmless.embodiments` — list the registered adapters."""

from harmless.registry import list_embodiments


def main():
    for name in list_embodiments():
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
