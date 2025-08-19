from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Scope(str, Enum):
    ONE_VIDEO = "one_video"
    SEED_PLUS_TAG = "seed_plus_tag"
    SEED_PLUS_CHANNEL = "seed_plus_channel"
    ANY = "any"

class Deps(BaseModel):

    # Scope / expansion policy
    scope: Scope = Scope.ANY
    allow_expand: bool = True  # tools may expand only if True

    # Reranking controls
    tag_rerank: bool = True
    rerank_alpha: float = 0.8   # weight for cosine similarity
    rerank_beta: float = 0.2    # weight for tag Jaccard overlap

    # Seed context (used by filters & rerank)
    seed_tags: List[str] = Field(default_factory=list)
    seed_video_id: str = ""       # the current seed video
    seed_channel_id: str = ""     # channel of the seed video

    # Optional allowlist of video_ids the retrieval is permitted to use
    # (used for SEED_PLUS_TAG to restrict to tag-discovered videos)
    allowed_video_ids: Optional[List[str]] = None