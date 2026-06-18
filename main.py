"""
LENS v3 — Backend Server
Face Recognition Engine: InsightFace (buffalo_l model)
99.86% accuracy | 512-dimension embeddings | Self-hosted
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from database import db
from face_engine import face_engine
from stream_processor import stream_processor
from websocket_manager import ws_manager

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LENS backend...")

    # Load InsightFace model
    face_engine.load_model()

    # Load all embeddings from Supabase into memory
    await face_engine.load_embeddings_from_db()

    # Start stream processor background task
    asyncio.create_task(stream_processor.run())

    logger.info("LENS backend ready.")
    yield

    # Shutdown
    stream_processor.stop()
    logger.info("LENS backend stopped.")


app = FastAPI(title="LENS Backend", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/api/health")
async def health():
    identity_count = len(face_engine.identity_index)
    camera_count = len(stream_processor.active_streams)
    return {
        "status": "ok",
        "model_loaded": face_engine.model_loaded,
        "model_name": "buffalo_l",
        "identity_count": identity_count,
        "active_streams": camera_count,
        "websocket_clients": len(ws_manager.connections),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await websocket.receive_text()
            message = json.loads(data)

            # Client can set search targets via websocket
            if message.get("type") == "set_targets":
                stream_processor.set_targets(message.get("identity_ids", []))
                await ws_manager.send_to(websocket, {
                    "type": "targets_updated",
                    "identity_ids": message.get("identity_ids", [])
                })

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ─────────────────────────────────────────────
# ENROLL A SINGLE IDENTITY
# ─────────────────────────────────────────────

@app.post("/api/enroll")
async def enroll_identity(
    full_name: str = Form(...),
    nin: str = Form(...),
    id_type: str = Form("NIN"),
    date_of_birth: str = Form(None),
    gender: str = Form(None),
    nationality: str = Form(None),
    group_tag: str = Form("public"),
    notes: str = Form(None),
    image: UploadFile = File(...),
):
    if not face_engine.model_loaded:
        raise HTTPException(503, "Face recognition model not loaded yet")

    # Read image bytes
    image_bytes = await image.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Could not decode image")

    # Detect faces
    faces = face_engine.app.get(frame)

    if len(faces) == 0:
        return JSONResponse(
            status_code=422,
            content={"success": False, "error": "No face detected in the image. Try a clearer photo."}
        )

    if len(faces) > 1:
        return JSONResponse(
            status_code=422,
            content={"success": False, "error": f"{len(faces)} faces detected. Please upload a photo with only one person."}
        )

    face = faces[0]
    embedding = face.embedding.tolist()  # 512-number array

    # Build identity record
    identity_id = str(uuid.uuid4())
    identity = {
        "id": identity_id,
        "full_name": full_name,
        "nin": nin,
        "id_type": id_type,
        "date_of_birth": date_of_birth,
        "gender": gender or (face_engine.get_gender(face.gender)),
        "nationality": nationality,
        "group_tag": group_tag,
        "notes": notes,
        "embedding": embedding,
        "embeddings_multi": [embedding],
        "age_estimate": int(face.age) if hasattr(face, "age") else None,
        "enrolled_at": datetime.utcnow().isoformat(),
        "is_active": True,
    }

    # Save to Supabase
    saved = await db.insert_identity(identity)
    if not saved:
        raise HTTPException(500, "Failed to save identity to database")

    # Add to in-memory index immediately (no restart needed)
    face_engine.add_to_index(identity_id, full_name, nin, group_tag, [embedding])

    logger.info(f"Enrolled: {full_name} ({nin})")

    return {
        "success": True,
        "identity_id": identity_id,
        "full_name": full_name,
        "nin": nin,
        "face_detected": True,
        "age_estimate": identity.get("age_estimate"),
        "gender": identity.get("gender"),
    }


# ─────────────────────────────────────────────
# ADD MORE ANGLES TO EXISTING IDENTITY
# ─────────────────────────────────────────────

@app.post("/api/enroll/{identity_id}/angles")
async def add_angles(identity_id: str, image: UploadFile = File(...)):
    if not face_engine.model_loaded:
        raise HTTPException(503, "Model not loaded")

    image_bytes = await image.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    faces = face_engine.app.get(frame)
    if len(faces) == 0:
        return JSONResponse(status_code=422, content={"success": False, "error": "No face detected"})

    new_embedding = faces[0].embedding.tolist()

    # Fetch existing and append
    identity = await db.get_identity(identity_id)
    if not identity:
        raise HTTPException(404, "Identity not found")

    existing = identity.get("embeddings_multi") or [identity["embedding"]]
    existing.append(new_embedding)

    await db.update_embeddings(identity_id, existing)
    face_engine.add_to_index(
        identity_id,
        identity["full_name"],
        identity["nin"],
        identity["group_tag"],
        existing
    )

    return {"success": True, "total_angles": len(existing)}


# ─────────────────────────────────────────────
# BULK ENROLL FROM CSV
# ─────────────────────────────────────────────

@app.post("/api/enroll/bulk")
async def bulk_enroll(csv_file: UploadFile = File(...)):
    import csv
    import io
    import httpx

    if not face_engine.model_loaded:
        raise HTTPException(503, "Model not loaded")

    content = await csv_file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))

    results = {"total": 0, "success": 0, "failed": 0, "failures": []}

    async with httpx.AsyncClient(timeout=30) as client:
        for row in reader:
            results["total"] += 1
            try:
                photo_url = row.get("photo_url", "").strip()
                if not photo_url:
                    raise ValueError("No photo_url provided")

                # Download photo
                resp = await client.get(photo_url)
                resp.raise_for_status()

                np_array = np.frombuffer(resp.content, np.uint8)
                frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                if frame is None:
                    raise ValueError("Could not decode image")

                faces = face_engine.app.get(frame)
                if len(faces) == 0:
                    raise ValueError("No face detected in photo")

                embedding = faces[0].embedding.tolist()
                identity_id = str(uuid.uuid4())

                identity = {
                    "id": identity_id,
                    "full_name": row.get("full_name", "").strip(),
                    "nin": row.get("nin", "").strip(),
                    "id_type": row.get("id_type", "NIN").strip(),
                    "date_of_birth": row.get("date_of_birth", "").strip() or None,
                    "gender": row.get("gender", "").strip() or None,
                    "nationality": row.get("nationality", "").strip() or None,
                    "group_tag": row.get("group_tag", "public").strip(),
                    "notes": row.get("notes", "").strip() or None,
                    "photo_url": photo_url,
                    "embedding": embedding,
                    "embeddings_multi": [embedding],
                    "enrolled_at": datetime.utcnow().isoformat(),
                    "is_active": True,
                }

                await db.insert_identity(identity)
                face_engine.add_to_index(identity_id, identity["full_name"], identity["nin"], identity["group_tag"], [embedding])

                results["success"] += 1
                logger.info(f"Bulk enrolled: {identity['full_name']}")

            except Exception as e:
                results["failed"] += 1
                results["failures"].append({
                    "row": row.get("full_name", f"Row {results['total']}"),
                    "error": str(e)
                })

    return results


# ─────────────────────────────────────────────
# IDENTIFY FACES IN AN IMAGE
# ─────────────────────────────────────────────

@app.post("/api/identify")
async def identify(image: UploadFile = File(...)):
    if not face_engine.model_loaded:
        raise HTTPException(503, "Model not loaded")

    image_bytes = await image.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Could not decode image")

    detections = face_engine.recognize(frame)
    return {"detections": detections, "count": len(detections)}


# ─────────────────────────────────────────────
# IDENTITY CRUD
# ─────────────────────────────────────────────

@app.get("/api/identities")
async def list_identities(search: str = None, group: str = None, limit: int = 100, offset: int = 0):
    identities = await db.list_identities(search=search, group=group, limit=limit, offset=offset)
    return {"identities": identities, "count": len(identities)}


@app.get("/api/identities/{identity_id}")
async def get_identity(identity_id: str):
    identity = await db.get_identity(identity_id)
    if not identity:
        raise HTTPException(404, "Identity not found")
    return identity


@app.put("/api/identities/{identity_id}")
async def update_identity(identity_id: str, data: dict):
    updated = await db.update_identity(identity_id, data)
    if not updated:
        raise HTTPException(404, "Identity not found")

    # Refresh in-memory index
    await face_engine.load_embeddings_from_db()
    return {"success": True}


@app.delete("/api/identities/{identity_id}")
async def delete_identity(identity_id: str):
    deleted = await db.delete_identity(identity_id)
    if not deleted:
        raise HTTPException(404, "Identity not found")

    face_engine.remove_from_index(identity_id)
    return {"success": True}


# ─────────────────────────────────────────────
# SEARCH TARGETS
# ─────────────────────────────────────────────

@app.post("/api/search/targets")
async def set_targets(data: dict):
    identity_ids = data.get("identity_ids", [])
    stream_processor.set_targets(identity_ids)

    # Notify all WebSocket clients
    await ws_manager.broadcast({
        "type": "targets_updated",
        "identity_ids": identity_ids
    })

    return {"success": True, "active_targets": identity_ids}


@app.delete("/api/search/targets")
async def clear_targets():
    stream_processor.set_targets([])
    await ws_manager.broadcast({"type": "targets_cleared"})
    return {"success": True}


# ─────────────────────────────────────────────
# CAMERAS
# ─────────────────────────────────────────────

@app.get("/api/cameras")
async def list_cameras():
    cameras = await db.list_cameras()
    # Attach live status from stream processor
    for cam in cameras:
        cam["is_live"] = cam["id"] in stream_processor.active_streams
    return {"cameras": cameras}


@app.post("/api/cameras")
async def add_camera(data: dict):
    camera_id = str(uuid.uuid4())
    rtmp_key = data.get("rtmp_key") or data["name"].lower().replace(" ", "_")

    camera = {
        "id": camera_id,
        "name": data["name"],
        "location": data.get("location", ""),
        "rtmp_key": rtmp_key,
        "stream_url": f"rtmp://0.0.0.0:1935/{rtmp_key}",
        "is_active": True,
        "added_at": datetime.utcnow().isoformat(),
    }

    saved = await db.insert_camera(camera)
    if not saved:
        raise HTTPException(500, "Failed to save camera")

    # Start processing this stream
    asyncio.create_task(stream_processor.add_stream(camera_id, rtmp_key))

    return {
        "success": True,
        "camera": camera,
        "rtmp_push_url": f"rtmp://YOUR_SERVER_IP:1935/{rtmp_key}",
        "hls_view_url": f"http://YOUR_SERVER_IP:8888/{rtmp_key}/index.m3u8",
    }


@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    stream_processor.remove_stream(camera_id)
    await db.delete_camera(camera_id)
    return {"success": True}


# ─────────────────────────────────────────────
# DETECTION LOGS
# ─────────────────────────────────────────────

@app.get("/api/logs")
async def get_logs(
    identity_id: str = None,
    camera_id: str = None,
    limit: int = 100,
    offset: int = 0
):
    logs = await db.get_logs(identity_id=identity_id, camera_id=camera_id, limit=limit, offset=offset)
    return {"logs": logs, "count": len(logs)}


@app.delete("/api/logs")
async def clear_logs():
    await db.clear_logs()
    return {"success": True}
