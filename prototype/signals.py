from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any

# ---------------------------------------
# OpenAI Vision Helper
# ---------------------------------------
import base64
from openai import OpenAI

import csv
import os
from functools import lru_cache

CSV_NAME_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "examples",
    "names_db.csv"
)

# EXPECTATION:
# The user will export:  export OPENAI_API_KEY="..."
client = OpenAI()


# ---------------------------------------
# CSV Name DB Loader
# ---------------------------------------
@lru_cache(maxsize=1)
def load_name_db() -> Dict[str, Dict[str, float]]:
    """Load the name-probability database from CSV (cached)."""
    db = {}

    if not os.path.exists(CSV_NAME_DB_PATH):
        print(f"[WARN] Name DB CSV not found: {CSV_NAME_DB_PATH}")
        return db

    with open(CSV_NAME_DB_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip().lower()
            db[name] = {
                "p_male": float(row["p_male"]),
                "p_female": float(row["p_female"]),
            }

    return db


# ---------------------------------------
# Image Encoding + Vision Model Call
# ---------------------------------------
def encode_image(photo_path: str) -> str:
    """Convert image file → base64 for OpenAI Vision."""
    with open(photo_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_openai_gender_model(photo_path: str) -> Dict[str, Any]:
    """Call OpenAI Vision (4o-mini) and return gender probabilities."""

    b64 = encode_image(photo_path)

    prompt = """
You are a classifier. Analyze the face in the image and respond ONLY in this JSON format:

{
  "p_male": float between 0 and 1,
  "p_female": float between 0 and 1,
  "quality": "high" | "medium" | "low",
  "notes": "short explanation"
}

Rules:
- p_male + p_female must equal 1.
- If multiple faces or low quality, set quality="low".
- If unsure, output 0.5 / 0.5.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    }
                ],
            },
        ],
        temperature=0,
    )

    import json
    raw = response.choices[0].message.content
    data = json.loads(raw)

    return {
        "p_male": float(data.get("p_male", 0.5)),
        "p_female": float(data.get("p_female", 0.5)),
        "quality": data.get("quality", "medium"),
        "meta": {"notes": data.get("notes", "")},
    }


# ---------------------------------------
# Signal Dataclass
# ---------------------------------------
@dataclass
class Signal:
    """Single signal about gender from one source."""

    source: str
    p_male: float
    p_female: float
    quality: str
    raw_value: Any = None
    meta: Dict[str, Any] = None

    def normalize(self) -> None:
        total = self.p_male + self.p_female
        if total <= 0:
            self.p_male = 0.5
            self.p_female = 0.5
        else:
            self.p_male /= total
            self.p_female /= total


# ---------------------------------------
# Name Signal
# ---------------------------------------
def is_ambiguous_name(p_male: float, p_female: float, threshold: float = 0.2) -> bool:
    return abs(p_male - p_female) < threshold


def get_name_signal(first_name: Optional[str], name_db: Optional[Dict[str, Dict[str, float]]] = None) -> Signal:
    if not first_name:
        return Signal(
            source="name",
            p_male=0.5,
            p_female=0.5,
            quality="low",
            raw_value=first_name,
            meta={"reason": "missing_name"},
        )

    if name_db is None:
        name_db = load_name_db()

    key = first_name.strip().lower()
    record = name_db.get(key)

    if record is None:
        return Signal(
            source="name",
            p_male=0.5,
            p_female=0.5,
            quality="low",
            raw_value=first_name,
            meta={"reason": "name_not_found"},
        )

    p_male = record["p_male"]
    p_female = record["p_female"]
    ambiguous = is_ambiguous_name(p_male, p_female)
    quality = "medium" if ambiguous else "high"

    return Signal(
        source="name",
        p_male=p_male,
        p_female=p_female,
        quality=quality,
        raw_value=first_name,
        meta={"ambiguous": ambiguous, "db_hit": True},
    )


# ---------------------------------------
# Sport Signal
# ---------------------------------------
def get_sport_signal(sport_gender: Optional[str]) -> Signal:
    if not sport_gender:
        return Signal(
            source="sport",
            p_male=0.5,
            p_female=0.5,
            quality="low",
            raw_value=sport_gender,
            meta={"reason": "missing_sport_gender"},
        )

    value = sport_gender.strip().lower()

    # NEW: neutral sport categories → neutral gender signal
    if value in {"unknown", "neutral", "coed", "mixed"}:
        return Signal(
            source="sport",
            p_male=0.5,
            p_female=0.5,
            quality="medium",
            raw_value=sport_gender,
            meta={"reason": "neutral_sport_category"},
        )

    # Masculine / feminine teams
    if value in {"male", "female"}:
        p_male = 1.0 if value == "male" else 0.0
        return Signal(
            source="sport",
            p_male=p_male,
            p_female=1 - p_male,
            quality="high",
            raw_value=sport_gender,
            meta={"reason": "single_gender_team"},
        )

    # fallback
    return Signal(
        source="sport",
        p_male=0.5,
        p_female=0.5,
        quality="low",
        raw_value=sport_gender,
        meta={"reason": "non_binary_or_mixed_team"},
    )


# ---------------------------------------
# Photo Signal
# ---------------------------------------
def get_photo_signal(photo_path: Optional[str], photo_model: Optional[Any] = None, is_group: bool = False) -> Signal:
    if not photo_path:
        return Signal(
            source="photo",
            p_male=0.5,
            p_female=0.5,
            quality="low",
            raw_value=photo_path,
            meta={"reason": "no_photo"},
        )

    # NEW: force neutral if group photo is flagged
    if is_group:
        return Signal(
            source="photo",
            p_male=0.5,
            p_female=0.5,
            quality="low",
            raw_value=photo_path,
            meta={"reason": "group_photo_detected"},
        )

    # Mocked model for testing
    if photo_model is not None:
        result = photo_model(photo_path)
        signal = Signal(
            source="photo",
            p_male=float(result.get("p_male", 0.5)),
            p_female=float(result.get("p_female", 0.5)),
            quality=result.get("quality", "medium"),
            raw_value=photo_path,
            meta=result.get("meta", {}),
        )
        signal.normalize()
        return signal

    # Real OpenAI Vision call
    try:
        result = call_openai_gender_model(photo_path)
        signal = Signal(
            source="photo",
            p_male=result["p_male"],
            p_female=result["p_female"],
            quality=result["quality"],
            raw_value=photo_path,
            meta=result["meta"],
        )
        signal.normalize()
        return signal

    except Exception as e:
        return Signal(
            source="photo",
            p_male=0.5,
            p_female=0.5,
            quality="low",
            raw_value=photo_path,
            meta={"error": str(e), "reason": "openai_call_failed"},
        )
