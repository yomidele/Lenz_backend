"""
LENS Face Engine
Uses InsightFace buffalo_l model
512-dimension embeddings, cosine similarity matching
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IndexedIdentity:
    identity_id: str
    full_name: str
    nin: str
    group_tag: str
    embeddings: List[np.ndarray]  # One or more 512-d vectors


class FaceEngine:
    def __init__(self):
        self.app = None
        self.model_loaded = False
        self.similarity_threshold = 0.45  # Tune between 0.40 and 0.55
        self.min_confidence = 60.0        # Don't label below this %
        self.identity_index: Dict[str, IndexedIdentity] = {}

    # ─────────────────────────────────────────────
    # MODEL LOADING
    # ─────────────────────────────────────────────

    def load_model(self):
        """Load InsightFace buffalo_l model into memory."""
        try:
            from insightface.app import FaceAnalysis
            logger.info("Loading InsightFace buffalo_l model...")

            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"]
                # Use CUDAExecutionProvider if you have a GPU:
                # providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.model_loaded = True
            logger.info("InsightFace model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            self.model_loaded = False

    # ─────────────────────────────────────────────
    # EMBEDDING INDEX (in-memory for fast matching)
    # ─────────────────────────────────────────────

    async def load_embeddings_from_db(self):
        """Load all active identity embeddings from Supabase into memory."""
        from database import db

        logger.info("Loading embeddings from database...")
        identities = await db.list_identities(limit=100000)
        self.identity_index = {}

        for identity in identities:
            embeddings_raw = identity.get("embeddings_multi") or []
            if not embeddings_raw and identity.get("embedding"):
                embeddings_raw = [identity["embedding"]]

            if not embeddings_raw:
                continue

            embeddings = [np.array(e, dtype=np.float32) for e in embeddings_raw]

            self.identity_index[identity["id"]] = IndexedIdentity(
                identity_id=identity["id"],
                full_name=identity["full_name"],
                nin=identity.get("nin", ""),
                group_tag=identity.get("group_tag", "public"),
                embeddings=embeddings,
            )

        logger.info(f"Loaded {len(self.identity_index)} identities into memory index.")

    def add_to_index(self, identity_id: str, full_name: str, nin: str, group_tag: str, embeddings_raw: list):
        """Add or update an identity in the in-memory index."""
        embeddings = [np.array(e, dtype=np.float32) for e in embeddings_raw]
        self.identity_index[identity_id] = IndexedIdentity(
            identity_id=identity_id,
            full_name=full_name,
            nin=nin,
            group_tag=group_tag,
            embeddings=embeddings,
        )

    def remove_from_index(self, identity_id: str):
        """Remove an identity from the in-memory index."""
        self.identity_index.pop(identity_id, None)

    # ─────────────────────────────────────────────
    # COSINE SIMILARITY
    # ─────────────────────────────────────────────

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compare two 512-d face embeddings. Returns 0.0 to 1.0."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def find_best_match(self, query_embedding: np.ndarray) -> Optional[dict]:
        """
        Compare query embedding against all stored identities.
        Returns the best match above threshold, or None if unknown.
        """
        best_score = 0.0
        best_identity = None

        for identity in self.identity_index.values():
            # Compare against all stored angles for this person
            for stored_embedding in identity.embeddings:
                score = self.cosine_similarity(query_embedding, stored_embedding)
                if score > best_score:
                    best_score = score
                    best_identity = identity

        confidence = best_score * 100.0

        if best_score >= self.similarity_threshold and confidence >= self.min_confidence:
            return {
                "identity_id": best_identity.identity_id,
                "full_name": best_identity.full_name,
                "nin": best_identity.nin,
                "group_tag": best_identity.group_tag,
                "confidence": round(confidence, 1),
                "is_unknown": False,
            }

        return None  # Unknown face

    # ─────────────────────────────────────────────
    # MAIN RECOGNITION FUNCTION
    # ─────────────────────────────────────────────

    def recognize(self, frame) -> list:
        """
        Run InsightFace on a frame.
        Returns list of detection dicts with bounding boxes and identity.
        """
        if not self.model_loaded or self.app is None:
            return []

        try:
            faces = self.app.get(frame)
        except Exception as e:
            logger.error(f"InsightFace detection error: {e}")
            return []

        detections = []

        for face in faces:
            # Bounding box
            box = face.bbox.astype(int)
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y

            # Face embedding (512 numbers)
            embedding = np.array(face.embedding, dtype=np.float32)

            # Match against database
            match = self.find_best_match(embedding)

            # Age and gender from InsightFace
            age = int(face.age) if hasattr(face, "age") and face.age is not None else None
            gender = self.get_gender(face.gender) if hasattr(face, "gender") else None

            if match:
                detection = {
                    "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "identity_id": match["identity_id"],
                    "full_name": match["full_name"],
                    "nin": match["nin"],
                    "confidence": match["confidence"],
                    "group_tag": match["group_tag"],
                    "age_estimate": age,
                    "gender": gender,
                    "is_unknown": False,
                    "is_target": False,  # Set by stream_processor if searching
                }
            else:
                detection = {
                    "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "identity_id": None,
                    "full_name": "UNIDENTIFIED",
                    "nin": None,
                    "confidence": 0.0,
                    "group_tag": "unknown",
                    "age_estimate": age,
                    "gender": gender,
                    "is_unknown": True,
                    "is_target": False,
                }

            detections.append(detection)

        return detections

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def get_gender(self, gender_value) -> str:
        """Convert InsightFace gender value to string."""
        if gender_value is None:
            return "Unknown"
        # InsightFace returns 1 for male, 0 for female
        if isinstance(gender_value, (int, float)):
            return "Male" if gender_value >= 0.5 else "Female"
        return str(gender_value)

    def set_threshold(self, threshold: float):
        """Update similarity threshold (0.35 to 0.60)."""
        self.similarity_threshold = max(0.35, min(0.60, threshold))

    def set_min_confidence(self, min_conf: float):
        """Update minimum confidence percentage to show a label."""
        self.min_confidence = max(40.0, min(95.0, min_conf))


# Singleton instance
face_engine = FaceEngine()
