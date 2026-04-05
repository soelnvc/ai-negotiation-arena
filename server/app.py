"""Root-level OpenEnv server entrypoint shim.

This delegates runtime to the canonical Startone FastAPI app.
"""

from startone.server.app import app, main as _startone_main


def main() -> None:
	"""Run the canonical Startone server entrypoint."""
	_startone_main()


if __name__ == "__main__":
	main()


__all__ = ["app", "main"]
