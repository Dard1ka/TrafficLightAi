import json, time, threading, asyncio, os
from typing import Dict, List, Any
import numpy as np
import cv2
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from yt_dlp import YoutubeDL
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# OpenCV/FFmpeg
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "reconnect;1|reconnect_streamed;1|reconnect_on_network_error;1|"
    "rw_timeout;15000000|http_persistent;0|multiple_requests;1"
)
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

# Configs
YOUTUBE_URL = "https://www.youtube.com/live/ByED80IKdIU"
CONF_THRESH = 0.25
IMGSZ = 640
INFER_EVERY_N_FRAMES = 1
AGGREGATE_EVERY_SEC = 5
BASE_GREEN = 20
EXTRA_MAX = 30
AUTO_MAX_PCU = True
LOCAL_STREAM = "http://127.0.0.1:18080/"

USE_SECTORS = False
USE_PRESET_POLY = True
ROI_JSON_PATH = "rois_4corners_poly.json"

PRESET_ROIS_NORM = [
    {"dir":"West","poly":[[0,0.85],[0.25,0.82],[0.2,0.67],[0,0.7]]},
    {"dir":"North","poly":[[0.24,0.57],[0.45,0.53],[0.12,0.11],[0.1,0.11]]},
    {"dir":"East","poly":[[0.75,0.6],[1,0.55],[1,0.47],[0.65,0.52]]},
    {"dir":"South","poly":[[0.57,0.79],[0.85,0.75],[1,0.9],[1,1],[0.71,1]]},
]

DIRECTIONS = ["West","North","East","South"]
VEHICLE_CLASSES = {"car","truck","bus","motorcycle","bicycle"}
PCU = {'motorcycle':0.5,'car':1.0,'kendaraan_besar':2.0,'bicycle':0.5}

# Visual
def draw_overlay(frame, rois_poly, dets):
    img = frame.copy()

    for r in rois_poly:
        pts = np.array(r["poly"], dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], isClosed=True, color=(0,255,255), thickness=2)
        x0,y0 = r["poly"][0]
        cv2.putText(img, r["dir"], (x0+6, y0+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

    for d in dets:
        x1,y1,x2,y2 = d["box_xyxy"]
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, d["label"], (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return img

last_frame_jpg: bytes | None = None
last_frame_lock = threading.Lock()

# Utils
def get_youtube_stream_url(u):
    with YoutubeDL({'quiet': True,'no_warnings': True,'format':'best'}) as ydl:
        return ydl.extract_info(u, download=False)['url']

def kategori_kendaraan(label):
    return "kendaraan_besar" if label in ("bus","truck") else label

def center_of_box(xyxy):
    x1,y1,x2,y2 = xyxy
    return (float(x1+x2)/2.0, float(y1+y2)/2.0)

def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        cond = ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1 + 1e-9) + x1)
        if cond: inside = not inside
    return inside

def build_fuzzy(max_pcu=60, extra_max=30):
    density = ctrl.Antecedent(np.linspace(0, max_pcu, 101), 'density')
    add     = ctrl.Consequent(np.linspace(0, extra_max, 101), 'add')
    mid = max_pcu/2
    density['low']  = fuzz.trimf(density.universe, [0, 0, 0.6*mid])
    density['med']  = fuzz.trimf(density.universe, [0.3*mid, mid, 0.8*max_pcu])
    density['high'] = fuzz.trimf(density.universe, [mid, max_pcu, max_pcu])
    add['small']  = fuzz.trimf(add.universe, [0,0,0.4*extra_max])
    add['medium'] = fuzz.trimf(add.universe, [0.2*extra_max,0.5*extra_max,0.8*extra_max])
    add['large']  = fuzz.trimf(add.universe, [0.6*extra_max,extra_max,extra_max])
    rules = [ctrl.Rule(density['low'],add['small']),
             ctrl.Rule(density['med'],add['medium']),
             ctrl.Rule(density['high'],add['large'])]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

def infer_extra_seconds(sim, v):
    sim.input['density'] = float(v); sim.compute(); return float(sim.output['add'])

def load_and_scale_rois(cap, path=ROI_JSON_PATH):
    ok, frame_probe = cap.read()
    if not ok:
        raise RuntimeError("Gagal baca frame untuk ukuran stream.")
    H, W = frame_probe.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if USE_PRESET_POLY and PRESET_ROIS_NORM:
        rois_scaled = []
        for r in PRESET_ROIS_NORM:
            poly = [[int(round(p[0]*W)), int(round(p[1]*H))] for p in r["poly"]]
            rois_scaled.append({"dir": r["dir"], "poly": poly})
        return rois_scaled, (W, H)

    with open(path, 'r') as f:
        data = json.load(f)
    rois_raw = data["rois"] if (isinstance(data, dict) and "rois" in data) else data

    rois_scaled = []
    for r in rois_raw:
        poly = r["poly"]
        is_norm = all(0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0 for p in poly)
        if is_norm:
            poly_scaled = [[int(round(p[0]*W)), int(round(p[1]*H))] for p in poly]
        else:
            poly_scaled = poly
        rois_scaled.append({"dir": r["dir"], "poly": poly_scaled})
    return rois_scaled, (W, H)

# Server State
class Hub:
    def __init__(self):
        self.clients: List[WebSocket] = []
        self.lock = asyncio.Lock()
        self.latest_packet: Dict[str, Any] = {}

    async def register(self, ws: WebSocket):
        await ws.accept()
        async with self.lock:
            self.clients.append(ws)

    async def unregister(self, ws: WebSocket):
        async with self.lock:
            if ws in self.clients:
                self.clients.remove(ws)

    async def broadcast(self, message: str):
        async with self.lock:
            dead = []
            for ws in self.clients:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.remove(ws)

hub = Hub()
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}

@app.get("/latest")
def latest():
    if hub.latest_packet:
        resp = JSONResponse(content=hub.latest_packet)
        resp.headers["Cache-Control"] = "no-store"
        return resp
    return Response(status_code=204, headers={"Cache-Control": "no-store"})

@app.get("/frame.jpg")
def frame_jpg():
    global last_frame_jpg
    with last_frame_lock:
        if last_frame_jpg is None:
            return Response(status_code=204, headers={"Cache-Control": "no-store"})
        return Response(content=last_frame_jpg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

@app.get("/viewer")
def viewer():
    html = r"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>AI Traffic Light – Dashboard</title>
      <style>
        :root { color-scheme: dark; }
        body { margin:0; background:#0f1115; color:#e8eaed; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",Arial,"Noto Sans","Apple Color Emoji","Segoe UI Emoji"; }
        header { padding:16px 20px; font-weight:700; font-size:20px; border-bottom:1px solid #222; }
        .wrap { display:grid; grid-template-columns: 1.5fr 1fr; gap:16px; padding:16px; }
        .card { background:#14181f; border:1px solid #222; border-radius:12px; overflow:hidden; }
        .card h3 { margin:0; padding:12px 14px; font-size:16px; border-bottom:1px solid #222; }
        .media { padding:12px; }
        .media img { width:100%; height:auto; display:block; border-radius:8px; }
        .json { padding:12px; }
        pre { margin:0; background:#0b0e13; border:1px solid #1c2230; border-radius:8px; padding:10px; overflow:auto; max-height:50vh; }
        table { width:100%; border-collapse:collapse; margin-top:12px; font-size:14px; }
        th, td { padding:8px 10px; border-bottom:1px solid #222; text-align:left; }
        th { position:sticky; top:0; background:#14181f; }
        .priority { background:#112a19; }
        .muted { color:#9aa0a6; font-size:12px; }
        .row { display:flex; align-items:center; gap:10px; padding:10px 12px; }
        .pill { padding:3px 8px; background:#1f2430; border:1px solid #2a3142; border-radius:999px; font-size:12px; }
        .ok    { color:#9ae6b4; }
        .warn  { color:#f6ad55; }
        .err   { color:#fc8181; }
      </style>
    </head>
    <body>
      <header>AI Traffic Light – Dashboard</header>

      <div class="wrap">
        <!-- kiri: gambar -->
        <div class="card">
          <h3>Live Overlay</h3>
          <div class="media">
            <img id="frame" alt="frame" src="/frame.jpg">
            <div class="muted">Sumber: <code>/frame.jpg</code> (auto‑refresh tiap 1 dtk)</div>
          </div>
        </div>

        <!-- kanan: JSON + tabel -->
        <div class="card">
          <div class="row">
            <h3 style="flex:1;margin:0;">Hasil Terakhir</h3>
            <span id="status" class="pill">memuat…</span>
          </div>
          <div class="json">
            <pre id="json">{}</pre>

            <table id="aggTable">
              <thead>
                <tr>
                  <th>Arah</th>
                  <th>PCU</th>
                  <th>car</th>
                  <th>motor</th>
                  <th>bike</th>
                  <th>besar</th>
                  <th>Extra(s)</th>
                  <th>Total Hijau(s)</th>
                  <th>Prioritas?</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>

            <div class="muted">JSON: <a href="/latest" target="_blank" style="color:#72a7ff">/latest</a></div>
          </div>
        </div>
      </div>

      <script>
        const img = document.getElementById('frame');
        const pre = document.getElementById('json');
        const tbody = document.querySelector('#aggTable tbody');
        const statusPill = document.getElementById('status');

        function setStatus(kind, text){
          statusPill.className = 'pill ' + (kind||'');
          statusPill.textContent = text;
        }

        function refreshImage(){
          img.src = '/frame.jpg?ts=' + Date.now();
        }

        function buildAggTable(data){
          const dirsOrder = ['West','North','East','South'];
          const agg = data.agg || {};
          const priority = data.priority || '';

          tbody.innerHTML = '';
          dirsOrder.forEach(dir=>{
            if(!(dir in agg)) return;
            const a = agg[dir];
            const tr = document.createElement('tr');
            if (dir === priority) tr.classList.add('priority');

            tr.innerHTML = `
              <td>${dir}</td>
              <td>${(a.pcu??0).toFixed(1)}</td>
              <td>${a.car??0}</td>
              <td>${a.motorcycle??0}</td>
              <td>${a.bicycle??0}</td>
              <td>${a.kendaraan_besar??0}</td>
              <td>${(a.extra??0).toFixed(1)}</td>
              <td>${(a.total_green??0).toFixed(1)}</td>
              <td>${dir===priority?'YA':'TIDAK'}</td>
            `;
            tbody.appendChild(tr);
          });
        }

        async function refreshJson(){
          try{
            const res = await fetch('/latest', {cache:'no-store'});
            if(res.status === 204){
              setStatus('warn','belum ada data');
              pre.textContent = '{}';
              tbody.innerHTML = '';
              return;
            }
            const j = await res.json();
            pre.textContent = JSON.stringify(j, null, 2);
            buildAggTable(j);
            setStatus('ok','ok');
          }catch(e){
            setStatus('err','gagal fetch');
          }
        }

        // auto refresh
        refreshImage(); refreshJson();
        setInterval(refreshImage, 1000);
        setInterval(refreshJson, 1000);
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await hub.register(ws)
    try:
        while True:
            await asyncio.sleep(30)
            await ws.send_text('{"type":"ping"}')
    except WebSocketDisconnect:
        await hub.unregister(ws)

# Main Detection Loop
def detection_loop():
    print("[loop] init model & stream ...")
    cap = cv2.VideoCapture(LOCAL_STREAM, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Gagal buka stream (cek Streamlink di port 18080).")

    ok, probe = cap.read()
    retry_read = 0
    while not ok and retry_read < 5:
        print("[loop] gagal baca frame awal, retry...")
        time.sleep(1)
        cap.release()
        cap = cv2.VideoCapture(LOCAL_STREAM, cv2.CAP_FFMPEG)
        ok, probe = cap.read()
        retry_read += 1
    if not ok:
        raise RuntimeError("Gagal baca frame awal setelah beberapa percobaan.")
    H, W = probe.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rois_poly, _ = load_and_scale_rois(cap, ROI_JSON_PATH)
    rois_norm = []
    for r in rois_poly:
        pts = []
        for x,y in r["poly"]:
            pts.append([x/float(W), y/float(H)])
        rois_norm.append({"dir": r["dir"], "poly": pts})

    model = YOLO("yolov8n.pt")

    unique_ids_this_window = set()
    agg_counts = {r["dir"]: {"kendaraan_besar":0, "car":0, "motorcycle":0, "bicycle":0} for r in rois_poly}
    agg_pcu = {r["dir"]:0.0 for r in rois_poly}

    frame_id=0
    t0=time.time()
    fuzzy_sim = build_fuzzy()

    last_dets = []
    fail_reads = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            fail_reads += 1
            print(f"[loop] read failed ({fail_reads}), reopen capture...")
            if fail_reads >= 5:
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(LOCAL_STREAM, cv2.CAP_FFMPEG)
                fail_reads = 0
            time.sleep(0.2)
            continue
        fail_reads = 0

        dets_json = []

        if frame_id % INFER_EVERY_N_FRAMES == 0:
            results = model.track(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF_THRESH,
                iou=0.5,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
                classes=[1, 2, 3, 5, 7],
            )
            r = results[0] if results else None
            if r is not None and r.boxes is not None and len(r.boxes) > 0:
                names = model.names
                for b in r.boxes:
                    if b.id is None:
                        continue
                    tid = int(b.id.item())
                    cls_name = names[int(b.cls.item())]
                    if cls_name not in VEHICLE_CLASSES:
                        continue

                    xyxy = b.xyxy.cpu().numpy().ravel()
                    cx, cy = center_of_box(xyxy)

                    hit_dir = None
                    for rroi in rois_poly:
                        if point_in_poly(cx, cy, rroi["poly"]):
                            hit_dir = rroi["dir"]
                            break
                    if hit_dir is None:
                        continue

                    kat = kategori_kendaraan(cls_name)

                    uid = f"{hit_dir}:{tid}"
                    if uid not in unique_ids_this_window:
                        unique_ids_this_window.add(uid)
                        agg_counts[hit_dir][kat] += 1
                        agg_pcu[hit_dir] += PCU.get(kat, 0.0)

                    x1,y1,x2,y2 = map(int, xyxy)
                    dets_json.append({
                        "label": kat,
                        "dir": hit_dir,
                        "box_xyxy": [x1,y1,x2,y2]
                    })

            last_dets = dets_json if dets_json else last_dets
        else:
            dets_json = last_dets
        try:
            draw = draw_overlay(frame, rois_poly, dets_json)
            ok_jpg, jpg = cv2.imencode('.jpg', draw, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok_jpg:
                global last_frame_jpg
                with last_frame_lock:
                    last_frame_jpg = jpg.tobytes()
        except Exception:
            pass

        if time.time()-t0 >= AGGREGATE_EVERY_SEC:
            pcu_vals = list(agg_pcu.values())
            pcu_max_seen = max(10.0, max(pcu_vals) if pcu_vals else 10.0)
            max_pcu_for_fuzzy = pcu_max_seen*1.2 if AUTO_MAX_PCU else 60
            fuzzy_sim = build_fuzzy(max_pcu_for_fuzzy, EXTRA_MAX)

            extra_dir, total_green = {}, {}
            for d in agg_pcu:
                extra = infer_extra_seconds(fuzzy_sim, agg_pcu[d])
                extra_dir[d] = float(extra)
                total_green[d] = float(BASE_GREEN + extra)

            if all(v == 0 for v in agg_pcu.values()):
                best_dir = None
            else:
                best_dir = max(agg_pcu, key=lambda k: agg_pcu[k]) if agg_pcu else None

            packet = {
                "ts": int(time.time()),
                "frame_size": {"w": int(W), "h": int(H)},
                "polys": rois_norm,
                "detections": last_dets,
                "agg": {
                    d: {
                        "pcu": float(agg_pcu[d]),
                        "car": int(agg_counts[d]["car"]),
                        "motorcycle": int(agg_counts[d]["motorcycle"]),
                        "bicycle": int(agg_counts[d]["bicycle"]),
                        "kendaraan_besar": int(agg_counts[d]["kendaraan_besar"]),
                        "extra": float(extra_dir.get(d,0.0)),
                        "total_green": float(total_green.get(d,BASE_GREEN)),
                    } for d in agg_pcu
                },
                "priority": best_dir if best_dir else ""
            }

            hub.latest_packet = packet
            msg = json.dumps(packet)
            try:
                asyncio.run(hub.broadcast(msg))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(hub.broadcast(msg))
                loop.close()

            unique_ids_this_window = set()
            agg_counts = {r["dir"]: {"kendaraan_besar":0, "car":0, "motorcycle":0, "bicycle":0} for r in rois_poly}
            agg_pcu = {r["dir"]:0.0 for r in rois_poly}
            t0=time.time()

        frame_id += 1

def start_bg():
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

start_bg()
