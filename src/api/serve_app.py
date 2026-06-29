import sys
import argparse

from src.api.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="LogSentinel API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-dir", default="models", help="Directory with trained models")
    args = parser.parse_args()

    import uvicorn
    app = create_app(model_dir=args.model_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
