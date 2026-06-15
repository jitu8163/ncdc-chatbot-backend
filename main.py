"""Convenience runner: `python main.py` starts the API with uvicorn.

For production use:  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
"""
import uvicorn


def main():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
