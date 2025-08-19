import os, time, uuid
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny
from models import Chunk


COLLECTION = "youtube_rag"
DIM = 3072

_client = None
def _make_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    print(f"[qdrant] connecting to {url}")
    return QdrantClient(url=url, api_key=api_key)

def _get_client() -> QdrantClient:
    global _client
    if not _client:
        _client = _make_client()
    return _client

def _wait_for_ready(timeout_s: int = 15):
    start = time.time()
    last_err = None
    print(f"[qdrant] waiting for readiness (timeout {timeout_s}s)...")
    while time.time() - start < timeout_s:
        try:
            _get_client().get_collections()
            print("[qdrant] ready")
            return
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(
        f"Qdrant not reachable at {os.getenv('QDRANT_URL','http://localhost:6333')}. "
        "Start Qdrant (docker) or set QDRANT_URL correctly."
    ) from last_err

def ensure_collection():
    _wait_for_ready()
    names = [c.name for c in _get_client().get_collections().collections]
    if COLLECTION not in names:
        print(f"[qdrant] creating collection '{COLLECTION}' size={DIM}")
        _get_client().recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )
    else:
        print(f"[qdrant] collection '{COLLECTION}' is ready")

def upsert_chunks(chunks: List[Chunk], vectors: List[List[float]]):
    ensure_collection()
    print(f"[qdrant] upserting {len(chunks)} chunks")
    points = []
    for ch, vec in zip(chunks, vectors):
        # Qdrant requires point IDs to be UUID or unsigned int. Use deterministic UUIDv5 as string.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"yt::{ch.video.video_id}::{ch.chunk_idx}"))
        payload = {
            "video_id": ch.video.video_id,
            "title": ch.video.title,
            "channel": ch.video.channel,
            "channel_id": ch.video.channel_id,
            "tags": ch.video.tags,
            "tag_set": ch.tag_set,
            "chunk_idx": ch.chunk_idx,
            "url": ch.video.url,
            "text": ch.text,
            "orig_id": ch.id,
        }
        points.append(PointStruct(id=point_id, vector=vec, payload=payload))
    _get_client().upsert(collection_name=COLLECTION, points=points)
    print("[qdrant] upsert complete")

def query(qvec: List[float], must_tags: Optional[List[str]] = None, top_k=8, query_filter: Optional[Filter]=None):
    ensure_collection()
    flt = query_filter
    if must_tags:
        tag_filter = Filter(must=[FieldCondition(key="tag_set", match=MatchAny(any=must_tags))])
        if flt:
            flt = Filter(must=(flt.must + tag_filter.must))
        else:
            flt = tag_filter
    print(f"[qdrant] query top_k={top_k} filtered={bool(must_tags)}")
    hits = _get_client().search(collection_name=COLLECTION, query_vector=qvec, limit=top_k, query_filter=flt)
    print(f"[qdrant] query hits={len(hits)}")
    return hits
