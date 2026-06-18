"""
LENS Stream Processor
Pulls frames from RTMP streams via FFmpeg
Runs InsightFace recognition on each frame
Broadcasts results via WebSocket
"""

import asyncio
import logging
import subprocess
import uuid
from datetime import datetime
from typing import Dict, Set

import numpy as np

logger = logging.getLogger(__name__)

# Frame dimensions — must match your camera output
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_CHANNELS = 3
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS

# How many frames per second to analyze
ANALYSIS_FPS = 2

# Minimum seconds between logging the same person on the same camera
LOG_COOLDOWN_SECONDS = 30


class StreamProcessor:
    def __init__(self):
        self.active_streams: Dict[str, dict] = {}   # camera_id → stream info
        self.target_ids: Set[str] = set()            # identity_ids being searched
        self._running = False
        self._log_cooldown: Dict[str, float] = {}   # "camera_id:identity_id" → last_log_time

    # ─────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────

    async def run(self):
        """Main loop — loads cameras from DB and starts processing."""
        self._running = True
        logger.info("Stream processor started.")

        from database import db
        cameras = await db.list_cameras()

        for camera in cameras:
            asyncio.create_task(
                self._process_stream(camera["id"], camera["rtmp_key"], camera["name"])
            )

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(5)

    def stop(self):
        self._running = False

    # ─────────────────────────────────────────────
    # STREAM MANAGEMENT
    # ─────────────────────────────────────────────

    async def add_stream(self, camera_id: str, rtmp_key: str, camera_name: str = "Camera"):
        """Start processing a new camera stream."""
        if camera_id not in self.active_streams:
            asyncio.create_task(
                self._process_stream(camera_id, rtmp_key, camera_name)
            )

    def remove_stream(self, camera_id: str):
        """Stop processing a camera stream."""
        if camera_id in self.active_streams:
            self.active_streams[camera_id]["active"] = False
            del self.active_streams[camera_id]
            logger.info(f"Stream removed: {camera_id}")

    def set_targets(self, identity_ids: list):
        """Set which identity IDs are being actively searched for."""
        self.target_ids = set(identity_ids)
        logger.info(f"Search targets updated: {self.target_ids}")

    # ─────────────────────────────────────────────
    # FRAME EXTRACTION + RECOGNITION LOOP
    # ─────────────────────────────────────────────

    async def _process_stream(self, camera_id: str, rtmp_key: str, camera_name: str):
        """
        Pull frames from an RTMP stream using FFmpeg.
        Run InsightFace recognition on each frame.
        Broadcast results via WebSocket.
        """
        from face_engine import face_engine
        from websocket_manager import ws_manager

        rtmp_url = f"rtmp://localhost:1935/{rtmp_key}"
        interval = 1.0 / ANALYSIS_FPS  # seconds between frame analyses

        self.active_streams[camera_id] = {
            "camera_id": camera_id,
            "rtmp_key": rtmp_key,
            "camera_name": camera_name,
            "active": True,
        }

        logger.info(f"Starting stream processor for: {rtmp_url}")

        # Notify dashboard this camera is being watched
        await ws_manager.broadcast({
            "type": "camera_status",
            "camera_id": camera_id,
            "camera_name": camera_name,
            "status": "connecting",
        })

        ffmpeg_cmd = [
            "ffmpeg",
            "-loglevel", "quiet",
            "-i", rtmp_url,
            "-vf", f"fps={ANALYSIS_FPS},scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]

        while self._running and self.active_streams.get(camera_id, {}).get("active"):
            process = None
            try:
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=FRAME_SIZE * 4,
                )

                await ws_manager.broadcast({
                    "type": "camera_status",
                    "camera_id": camera_id,
                    "camera_name": camera_name,
                    "status": "live",
                })

                logger.info(f"Stream live: {camera_name} ({rtmp_key})")

                while self._running and self.active_streams.get(camera_id, {}).get("active"):
                    raw = process.stdout.read(FRAME_SIZE)

                    if len(raw) < FRAME_SIZE:
                        logger.warning(f"Stream ended or frame too small: {camera_name}")
                        break

                    # Decode frame
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS))

                    # Run recognition (offload to thread so we don't block async loop)
                    detections = await asyncio.get_event_loop().run_in_executor(
                        None, face_engine.recognize, frame.copy()
                    )

                    # Mark targets
                    for det in detections:
                        if det.get("identity_id") in self.target_ids:
                            det["is_target"] = True

                    # Broadcast to dashboard
                    if detections:
                        await ws_manager.broadcast({
                            "type": "detection",
                            "camera_id": camera_id,
                            "camera_name": camera_name,
                            "timestamp": datetime.utcnow().isoformat(),
                            "detections": detections,
                        })

                        # Log known faces to DB (with cooldown)
                        for det in detections:
                            if not det["is_unknown"]:
                                await self._log_detection(det, camera_id, camera_name)

                            # Alert if target found
                            if det.get("is_target"):
                                await ws_manager.broadcast({
                                    "type": "target_found",
                                    "camera_id": camera_id,
                                    "camera_name": camera_name,
                                    "identity": {
                                        "name": det["full_name"],
                                        "nin": det["nin"],
                                        "confidence": det["confidence"],
                                    },
                                    "timestamp": datetime.utcnow().isoformat(),
                                })

                    await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Stream error ({camera_name}): {e}")

            finally:
                if process:
                    process.kill()
                    process.wait()

            # Stream went offline — notify dashboard
            await ws_manager.broadcast({
                "type": "camera_status",
                "camera_id": camera_id,
                "camera_name": camera_name,
                "status": "offline",
            })

            # Wait before trying to reconnect
            logger.info(f"Reconnecting to {camera_name} in 5s...")
            await asyncio.sleep(5)

        logger.info(f"Stream processor stopped: {camera_name}")

    # ─────────────────────────────────────────────
    # DETECTION LOGGING (with cooldown)
    # ─────────────────────────────────────────────

    async def _log_detection(self, detection: dict, camera_id: str, camera_name: str):
        """Log a detection to Supabase, respecting cooldown per person per camera."""
        from database import db
        import time

        cooldown_key = f"{camera_id}:{detection['identity_id']}"
        last_log = self._log_cooldown.get(cooldown_key, 0)

        if time.time() - last_log < LOG_COOLDOWN_SECONDS:
            return  # Too soon to log again

        self._log_cooldown[cooldown_key] = time.time()

        log = {
            "id": str(uuid.uuid4()),
            "identity_id": detection["identity_id"],
            "full_name": detection["full_name"],
            "nin": detection.get("nin"),
            "confidence": detection["confidence"],
            "camera_id": camera_id,
            "camera_name": camera_name,
            "age_estimate": detection.get("age_estimate"),
            "gender": detection.get("gender"),
            "detected_at": datetime.utcnow().isoformat(),
        }

        await db.insert_log(log)


# Singleton instance
stream_processor = StreamProcessor()
