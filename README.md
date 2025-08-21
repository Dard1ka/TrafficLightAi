# TrafficLightAi
This project uses YOLO (Ultralytics) for vehicle detection from YouTube streams/local streams, as well as Fuzzy Logic (scikit-fuzzy) for traffic density analysis.

There are two ways to run this project:
📓 Mode 1 — Jupyter Notebook / Google Colab (Without Local Server)

Use the requirements-For-ipynbOnly(NotLocalServer).txt file.
Steps:
1. Create a new environment.
2. Install dependencies:
   ```bash
   pip install -r requirements-For-ipynbOnly(NotLocalServer).txt
   ```
3. Export cookies from YouTube
   - Install the Get cookies.txt extension by Cookie-Editor.
   - Open youtube.com, then export cookies.
   - Save the cookies file (JSON format) in the same folder as .ipynb.

4. Run the notebook in Jupyter/VSCode Or upload it to Google Colab and immediately select Run all.
   
🖥️ Mode 2 — Local Server (Streamlink + FastAPI)
Use the requirements-With-Local_Server.txt file.

Steps:
1. Create a new environment (e.g., .server).
2. Install dependencies:
   ```bash
   pip install -r requirements-With-Local_Server.txt
   ```
3. Open two terminals / PowerShell.
4. Activate the environment in both terminals:
   ```bash
   .\.server\Scripts\Activate.ps1
   ```
5. Run Streamlink in the first terminal:
   ```bash
   streamlink --player-external-http --player-external-http-port 18080 "https://www.youtube.com/live/ByED80IKdIU" 480p
   ```
6. Run the server on the second terminal:
   ```bash
    uvicorn server:app --host 0.0.0.0 --port 8000
   ```
7. See the results in your browser:
   http://127.0.0.1:8000/viewer 

📂 Struktur File
```pqsql
project/
│── server.py                 # Server FastAPI
│── notebook.ipynb            # Notebook versi Colab/Jupyter
│── requirements-For-ipynbOnly(NotLocalServer).txt
│── requirements-With-Local_Server.txt
│── cookies.json              # (Hasil ekspor dari YouTube)

```
