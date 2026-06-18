"""
LENS WebSocket Manager
Manages all connected dashboard clients
Broadcasts real-time recognition results
"""

import json
import logging
from typing import List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.connections)}")

    async def broadcast(self, message: dict):
        """Send a message to ALL connected dashboard clients."""
        if not self.connections:
            return

        payload = json.dumps(message)
        dead = []

        for ws in self.connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        # Clean up dead connections
        for ws in dead:
            self.disconnect(ws)

    async def send_to(self, websocket: WebSocket, message: dict):
        """Send a message to ONE specific client."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"send_to error: {e}")
            self.disconnect(websocket)


# Singleton instance
ws_manager = WebSocketManager()
