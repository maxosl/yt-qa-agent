from pydantic import BaseModel, Field
from typing import List, Optional

class VideoMeta(BaseModel):
    video_id: str
    title: str
    channel: Optional[str] = None
    channel_id: Optional[str] = None  # REQUIRED for same-channel search
    tags: List[str] = Field(default_factory=list)
    url: str

class Chunk(BaseModel):
    id: str
    video: VideoMeta
    chunk_idx: int
    text: str
    tag_set: List[str]
