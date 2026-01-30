import subprocess
import threading
import webbrowser
import time
import os

# --- Récupérer le chemin du dossier /font ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Lancement du backend FastAPI ---
def run_backend():
    subprocess.run([
        "python", "-m", "uvicorn",
        "app:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload"
    ], cwd=API_DIR)


# --- Lancement du frontend Gradio ---
def run_frontend():
    time.sleep(9)  # Attendre que le backend démarre
    webbrowser.open("http://localhost:7860")

    # Exécuter frontend.py depuis le dossier /font/api/
    subprocess.run(["python", os.path.join(API_DIR, "frontend.py")])

# --- Threads pour exécution parallèle ---
if __name__ == "__main__":
    t1 = threading.Thread(target=run_backend)
    t2 = threading.Thread(target=run_frontend)

    t1.start()
    t2.start()

    t1.join()
    t2.join()
