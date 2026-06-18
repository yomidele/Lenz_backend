"""
LENS Database Layer
Supabase client — all DB operations go through here
"""

import logging
import os
from typing import Optional

from supabase import create_client, Client

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.client: Optional[Client] = None
        self._connect()

    def _connect(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            logger.error("SUPABASE_URL or SUPABASE_KEY not set in environment")
            return
        try:
            self.client = create_client(url, key)
            logger.info("Connected to Supabase.")
        except Exception as e:
            logger.error(f"Supabase connection failed: {e}")

    # ─────────────────────────────────────────────
    # IDENTITIES
    # ─────────────────────────────────────────────

    async def insert_identity(self, identity: dict) -> bool:
        try:
            self.client.table("identities").insert(identity).execute()
            return True
        except Exception as e:
            logger.error(f"insert_identity error: {e}")
            return False

    async def get_identity(self, identity_id: str) -> Optional[dict]:
        try:
            res = self.client.table("identities").select("*").eq("id", identity_id).single().execute()
            return res.data
        except Exception as e:
            logger.error(f"get_identity error: {e}")
            return None

    async def list_identities(
        self,
        search: str = None,
        group: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> list:
        try:
            query = self.client.table("identities").select("*").eq("is_active", True)

            if search:
                # Search by name OR nin
                query = query.or_(f"full_name.ilike.%{search}%,nin.ilike.%{search}%")

            if group:
                query = query.eq("group_tag", group)

            res = query.order("enrolled_at", desc=True).range(offset, offset + limit - 1).execute()
            return res.data or []
        except Exception as e:
            logger.error(f"list_identities error: {e}")
            return []

    async def update_identity(self, identity_id: str, data: dict) -> bool:
        try:
            # Never allow overwriting id or embeddings via this route
            safe_fields = {k: v for k, v in data.items() if k not in ("id", "embedding", "embeddings_multi")}
            self.client.table("identities").update(safe_fields).eq("id", identity_id).execute()
            return True
        except Exception as e:
            logger.error(f"update_identity error: {e}")
            return False

    async def update_embeddings(self, identity_id: str, embeddings_multi: list) -> bool:
        try:
            self.client.table("identities").update({
                "embeddings_multi": embeddings_multi,
                "embedding": embeddings_multi[0] if embeddings_multi else None
            }).eq("id", identity_id).execute()
            return True
        except Exception as e:
            logger.error(f"update_embeddings error: {e}")
            return False

    async def delete_identity(self, identity_id: str) -> bool:
        try:
            # Soft delete — set is_active to false
            self.client.table("identities").update({"is_active": False}).eq("id", identity_id).execute()
            return True
        except Exception as e:
            logger.error(f"delete_identity error: {e}")
            return False

    # ─────────────────────────────────────────────
    # CAMERAS
    # ─────────────────────────────────────────────

    async def insert_camera(self, camera: dict) -> bool:
        try:
            self.client.table("cameras").insert(camera).execute()
            return True
        except Exception as e:
            logger.error(f"insert_camera error: {e}")
            return False

    async def list_cameras(self) -> list:
        try:
            res = self.client.table("cameras").select("*").eq("is_active", True).execute()
            return res.data or []
        except Exception as e:
            logger.error(f"list_cameras error: {e}")
            return []

    async def delete_camera(self, camera_id: str) -> bool:
        try:
            self.client.table("cameras").update({"is_active": False}).eq("id", camera_id).execute()
            return True
        except Exception as e:
            logger.error(f"delete_camera error: {e}")
            return False

    # ─────────────────────────────────────────────
    # DETECTION LOGS
    # ─────────────────────────────────────────────

    async def insert_log(self, log: dict) -> bool:
        try:
            self.client.table("detection_logs").insert(log).execute()
            return True
        except Exception as e:
            logger.error(f"insert_log error: {e}")
            return False

    async def get_logs(
        self,
        identity_id: str = None,
        camera_id: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> list:
        try:
            query = self.client.table("detection_logs").select("*")

            if identity_id:
                query = query.eq("identity_id", identity_id)
            if camera_id:
                query = query.eq("camera_id", camera_id)

            res = query.order("detected_at", desc=True).range(offset, offset + limit - 1).execute()
            return res.data or []
        except Exception as e:
            logger.error(f"get_logs error: {e}")
            return []

    async def clear_logs(self) -> bool:
        try:
            self.client.table("detection_logs").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            return True
        except Exception as e:
            logger.error(f"clear_logs error: {e}")
            return False


# Singleton instance
db = Database()
