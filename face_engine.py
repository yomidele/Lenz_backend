"""
LENS Face Engine
Uses InsightFace buffalo_sc model (lightweight, fits Railway free tier)
~300MB RAM | 97% accuracy | 512-dimension embeddings | cosine similarity
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IndexedIdentity:
    identity_id: str
    full_name: str
    nin: str
    group_tag: str
    embeddings: List[np.ndarray]


class FaceEngine:
    def __init__(self):
        self.app = None
        self.model_loaded = False
        self.similarity_threshold = 0.45
        self.min_confidence = 60.0
        self.identity_index: Dict[str, IndexedIdentity] = {}

    # ─────────────────────────────────────────────
    # MODEL LOADING
    # ─────────────────────────────────────────────

    def load_model(self):
        try:
            from insightface.app import FaceAnalysis
            logger.info("Loading InsightFace buffalo_sc model...")

            self.app = FaceAnalysis(
                name="buffalo_sc",
                providers=["CPUExecutionProvider"]
            )
            # Smaller det_size = less RAM used
            self.app.prepare(ctx_id=0, det_size=(320, 320))
            self.model_loaded = True
            logger.info("InsightFace buffalo_sc model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            self.model_loaded = False

    # ─────────────────────────────────────────────
    # EMBEDDING INDEX
    # ─────────────────────────────────────────────

    async def load_embeddings_from_db(self):
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

            embeddings = [
                np.array(e, dtype=np.float32)
                for e in embeddings_raw
            ]

            self.identity_index[identity["id"]] = IndexedIdentity(
                identity_id=identity["id"],
                full_name=identity["full_name"],
                nin=identity.get("nin", ""),
                group_tag=identity.get("group_tag", "public"),
                embeddings=embeddings,
            )

        logger.info(
            f"Loaded {len(self.identity_index)} identities into memory."
        )

    def add_to_index(
        self,
        identity_id: str,
        full_name: str,
        nin: str,
        group_tag: str,
        embeddings_raw: list
    ):
        embeddings = [
            np.array(e, dtype=np.float32)
            for e in embeddings_raw
        ]
        self.identity_index[identity_id] = IndexedIdentity(
            identity_id=identity_id,
            full_name=full_name,
            nin=nin,
            group_tag=group_tag,
            embeddings=embeddings,
        )

    def remove_from_index(self, identity_id: str):
        self.identity_index.pop(identity_id, None)

    # ─────────────────────────────────────────────
    # COSINE SIMILARITY MATCHING
    # ─────────────────────────────────────────────

    def cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def find_best_match(
        self,
        query_embedding: np.ndarray
    ) -> Optional[dict]:
        best_score = 0.0
        best_identity = None

        for identity in self.identity_index.values():
            for stored_embedding in identity.embeddings:
                score = self.cosine_similarity(
                    query_embedding,
                    stored_embedding
                )
                if score > best_score:
                    best_score = score
                    best_identity = identity

        confidence = best_score * 100.0

        if (
            best_score >= self.similarity_threshold
            and confidence >= self.min_confidence
        ):
            return {
                "identity_id": best_identity.identity_id,
                "full_name": best_identity.full_name,
                "nin": best_identity.nin,
                "group_tag": best_identity.group_tag,
                "confidence": round(confidence, 1),
                "is_unknown": False,
            }

        return None

    # ─────────────────────────────────────────────
    # MAIN RECOGNITION
    # ─────────────────────────────────────────────

    def recognize(self, frame) -> list:
        if not self.model_loaded or self.app is None:
            return []

        try:
            faces = self.app.get(frame)
        except Exception as e:
            logger.error(f"InsightFace detection error: {e}")
            return []

        detections = []

        for face in faces:
            box = face.bbox.astype(int)
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y

            embedding = np.array(
                face.embedding,
                dtype=np.float32
            )

            match = self.find_best_match(embedding)

            age = (
                int(face.age)
                if hasattr(face, "age") and face.age is not None
                else None
            )
            gender = (
                self.get_gender(face.gender)
                if hasattr(face, "gender")
                else None
            )

            if match:
                detection = {
                    "box": {
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h)
                    },
                    "identity_id": match["identity_id"],
                    "full_name": match["full_name"],
                    "nin": match["nin"],
                    "confidence": match["confidence"],
                    "group_tag": match["group_tag"],
                    "age_estimate": age,
                    "gender": gender,
                    "is_unknown": False,
                    "is_target": False,
                }
            else:
                detection = {
                    "box": {
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h)
                    },
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
        if gender_value is None:
            return "Unknown"
        if isinstance(gender_value, (int, float)):
            return "Male" if gender_value >= 0.5 else "Female"
        return str(gender_value)

    def set_threshold(self, threshold: float):
        self.similarity_threshold = max(0.35, min(0.60, threshold))

    def set_min_confidence(self, min_conf: float):
        self.min_confidence = max(40.0, min(95.0, min_conf))


# Singleton instance
face_engine = FaceEngine()
