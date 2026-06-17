import os
import socket
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def check_dependencies():
    print("Checking dependencies...")
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401

        print("Core dependencies found.")
    except ImportError:
        print("Missing required dependencies. Installing from backend/requirements.txt ...")
        req_file = os.path.join(PROJECT_ROOT, "backend", "requirements.txt")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Dependency installation failed: {e}")
            print("Try running manually:")
            print(f'  "{sys.executable}" -m pip install -r "{req_file}"')
            sys.exit(1)
        print("Dependencies installed successfully.")


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) == 0


def run_server():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))

    print(
        """
    ===================================================
    DEEPFAKE SENTINEL - DETECTION PLATFORM
    ===================================================
    """
    )

    if port_in_use(host, port):
        print(f"[ERROR] Port {port} is already in use on {host}.")
        print("Set a different port via:  PORT=8080 python run.py")
        sys.exit(1)

    print(f"The Dashboard will be available at: http://{host}:{port}\n")

    os.environ["PYTHONPATH"] = PROJECT_ROOT

    try:
        import uvicorn

        uvicorn.run("backend.main:app", host=host, port=port, reload=True)
    except KeyboardInterrupt:
        print("\nShutting down DeepFake Sentinel. Goodbye!")
    except Exception as e:
        print(f"\n[ERROR] Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    check_dependencies()
    run_server()
