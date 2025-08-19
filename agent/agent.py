import os
from typing import List, Optional
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext
from pydantic_ai.models.openai import OpenAIModel

from models import VideoMeta, Chunk, Deps, Scope
from services import (get_transcript_text, get_video_meta, normalize_tags, chunk_text, 
    embed_texts, upsert_chunks, search_videos_by_tags, discover_by_tags_and_channel,
    list_channel_uploads, query)
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue


SYSTEM_PROMPT = """You answer questions about one or more YouTube videos.
Scope rules are enforced by the system (retrieval filters) and provided via deps.scope.
Do not request expansions outside scope. If scope is ONE_VIDEO, do not expand; if SEED_PLUS_TAG,
you may expand only by tag search; if SEED_PLUS_CHANNEL, only by same-channel uploads; if ANY, you may use both.
Use retrieved chunks and cite titles with links. If insufficient evidence, say so."""


MODEL_NAME = "gpt-5"
rag_agent = Agent(
    model=OpenAIModel(MODEL_NAME),
    system_prompt=SYSTEM_PROMPT,
    deps_type=Deps,
)

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

# ----- Pure, procedural functions -----
async def index_video_id_plain(video_id: str) -> VideoMeta:
    print(f"[index] fetching meta for video_id={video_id}")
    meta = get_video_meta(video_id)
    text = get_transcript_text(video_id)
    if not text:
        print(f"[index] no transcript for {video_id}; skipping chunk/embeddings")
        return meta
    print(f"[index] transcript length={len(text)} chars for {video_id}")
    chunks = [
        Chunk(
            id=f"{meta.video_id}#{idx}",
            video=meta,
            chunk_idx=idx,
            text=s,
            tag_set=normalize_tags(meta.tags),
        )
        for idx, s in enumerate(chunk_text(text))
    ]
    print(f"[index] chunked into {len(chunks)} chunks for {video_id}")
    if chunks:
        print(f"[index] embedding {len(chunks)} chunks for {video_id}")
        vectors = embed_texts([c.text for c in chunks])
        upsert_chunks(chunks, vectors)
        print(f"[index] upserted {len(chunks)} chunks to vector store for {video_id}")
    return meta

async def expand_hybrid_plain(seed_video_id: str, seed_tags: List[str],
                              per_tag: int = 6, channel_max: int = 25) -> List[str]:
    # Fetch correct channel_id internally (prevents arg mixups)
    print(f"[expand] seed={seed_video_id} per_tag={per_tag} channel_max={channel_max}")
    meta = get_video_meta(seed_video_id)
    vids = discover_by_tags_and_channel(seed_video_id, seed_tags, meta.channel_id or "", per_tag, channel_max)
    print(f"[expand] discovered {len(vids)} videos")
    for vid in vids:
        try:
            print(f"[expand] indexing discovered video {vid}")
            await index_video_id_plain(vid)
        except Exception as e:
            print(f"[expand] failed to index {vid}: {e}")
            pass
    return vids


async def expand_scoped_plain(scope: Scope, seed_video_id: str, seed_tags: List[str],
                              seed_channel_id: str, per_tag: int = 5, channel_max: int = 25) -> List[str]:
    vids: List[str] = []
    if scope == Scope.ONE_VIDEO:
        return vids
    if scope in (Scope.SEED_PLUS_CHANNEL, Scope.ANY):
        # cheap path: uploads playlist
        for vid in list_channel_uploads(seed_channel_id, max_results=channel_max):
            if vid != seed_video_id:
                vids.append(vid)
    if scope in (Scope.SEED_PLUS_TAG, Scope.ANY):
        for vid in search_videos_by_tags(seed_tags, max_per_tag=per_tag):
            if vid != seed_video_id:
                vids.append(vid)
    # dedupe while preserving order
    seen, ordered = set(), []
    for v in vids:
        if v not in seen:
            seen.add(v); ordered.append(v)
    # index
    for vid in ordered:
        try:
            await index_video_id_plain(vid)
        except Exception:
            pass
    return ordered

def build_scope_filter(scope: Scope, seed_video_id: str, seed_channel_id: str,
                       allowed_video_ids: Optional[List[str]] = None) -> Optional[Filter]:
    if scope == Scope.ONE_VIDEO:
        return Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=seed_video_id))])
    if scope == Scope.SEED_PLUS_CHANNEL:
        return Filter(must=[FieldCondition(key="channel_id", match=MatchValue(value=seed_channel_id))])
    if scope == Scope.SEED_PLUS_TAG:
        # If you want to restrict to *only* the tag-expanded set, pass allowed_video_ids
        if allowed_video_ids:
            return Filter(must=[FieldCondition(key="video_id", match=MatchAny(any=allowed_video_ids))])
        # If not provided, fall back to no hard filter (semantic + tag rerank still biases)
        return None
    return None  # ANY

# ----- Tools (used by the agent when it decides to) -----
@rag_agent.tool
async def index_by_id(ctx: RunContext[Deps], video_id: str) -> VideoMeta:
    print(f"[tool.index_by_id] video_id={video_id}")
    return await index_video_id_plain(video_id)

@rag_agent.tool
async def expand_hybrid(ctx: RunContext[Deps], per_tag: int = 6, related_max: int = 25) -> List[str]:
    seed_video_id = ctx.deps.seed_video_id
    seed_tags = ctx.deps.seed_tags
    if not ctx.deps.allow_expand:
        return []
    print(f"[tool.expand_hybrid] seed={seed_video_id} per_tag={per_tag} related_max={related_max} allow_expand={ctx.deps.allow_expand}")
    return await expand_scoped_plain(ctx.deps.scope, seed_video_id, seed_tags, ctx.deps.seed_channel_id,
                                     per_tag=per_tag, channel_max=related_max)

@rag_agent.tool
async def rag_search(ctx: RunContext[Deps], query_text: str, prefer_tags: Optional[List[str]] = None, k: int = 8) -> List[dict]:
    print(f"[tool.rag_search] q='{query_text[:60]}...' k={k}")
    qvec = embed_texts([query_text])[0]
    scope_filter = build_scope_filter(ctx.deps.scope, ctx.deps.seed_video_id, ctx.deps.seed_channel_id,
                                      ctx.deps.allowed_video_ids)
    hits = query(qvec, must_tags=None, top_k=k, query_filter=scope_filter)    
    print(f"[tool.rag_search] raw hits={len(hits)}")

    results = [h.payload | {"cosine": h.score} for h in hits]

    # --- Tag-aware re-rank ---
    if ctx.deps.tag_rerank:
        ref_tags = prefer_tags or ctx.deps.seed_tags or []
        alpha, beta = ctx.deps.rerank_alpha, ctx.deps.rerank_beta
        for r in results:
            overlap = _jaccard(r.get("tag_set") or [], ref_tags)
            r["combined_score"] = alpha * float(r["cosine"]) + beta * overlap
        results.sort(key=lambda r: r.get("combined_score", r["cosine"]), reverse=True)
        print(f"[tool.rag_search] reranked with alpha={alpha} beta={beta} ref_tags={len(ref_tags)}")

    # Return top-k (already small) with scores included
    print(f"[tool.rag_search] returning {min(len(results), k)} results")
    return results
