import asyncio
import sys

from dotenv import load_dotenv
load_dotenv()

from agent import expand_scoped_plain, Deps, index_video_id_plain, Scope, rag_agent
from services import infer_scope_llm 



async def ingest_seed_video(video_id: str):
    print(f"[ingest] start video_id={video_id}")
    await index_video_id_plain(video_id)


async def answer(video_id: str, question: str) -> str:
    # Index the seed (idempotent) and optionally expand the corpus up front
    print(f"[answer] start video_id={video_id} question_len={len(question)}")
    meta = await index_video_id_plain(video_id)
    # LLM infers scope
    print(f"[answer] inferring scope")
    scope_str, rationale = infer_scope_llm(question, seed_title=meta.title, seed_tags=meta.tags)
    scope = Scope(scope_str)
    print(f"[answer] inferred scope: {scope} with reason: {rationale}")
    # Expand only within inferred scope (deterministic pre-expansion)
    expanded_ids = await expand_scoped_plain(
        scope, seed_video_id=video_id, seed_tags=meta.tags, seed_channel_id=meta.channel_id or "",
        per_tag=5, channel_max=25
    )

     # Build the allowlist used by retrieval, per scope
    allowed_video_ids = None
    if scope == Scope.ONE_VIDEO:
        allowed_video_ids = [video_id]
    elif scope == Scope.SEED_PLUS_TAG:
        # seed + tag-discovered only (exclude other channels not in tag set)
        allowed_video_ids = [video_id] + expanded_ids
    elif scope == Scope.SEED_PLUS_CHANNEL:
        # we’ll use channel_id filter; no need for explicit allowlist
        allowed_video_ids = None
    else:
        # ANY => no hard restriction
        allowed_video_ids = None

    # Pass seed tags + rerank knobs to the agent
    print("[answer] running agent...")
    run = await rag_agent.run(
        f"Question: {question}\nSeed video: {video_id}",
        deps=Deps(
            scope=scope,
            allow_expand=False,
            tag_rerank=True,
            rerank_alpha=0.8,
            rerank_beta=0.2,
            seed_tags=meta.tags,
            seed_video_id=video_id,
            seed_channel_id=meta.channel_id or "",
            allowed_video_ids=allowed_video_ids,  # may be None
        ),
    )
    print(f"[answer] agent completed, output_chars={len(run.output)}")
    return run.output

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app.py <VIDEO_ID> [question ...]")
        sys.exit(1)
    vid = sys.argv[1]
    q = " ".join(sys.argv[2:]) or "Summarize this video and related context."
    print(f"[main] video_id={vid}")
    print(f"[main] question_len={len(q)}")
    asyncio.run(ingest_seed_video(vid))
    print(asyncio.run(answer(vid, q)))