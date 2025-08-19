import re
import time
import json
import os
import threading
from typing import Iterable, List
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from googleapiclient.errors import HttpError

from googleapiclient.discovery import build
from models import VideoMeta

YOUTUBE_WATCH_URL_TEMPLATE = "https://www.youtube.com/watch?v={video_id}"
CHANNEL_ID_RE = re.compile(r"^UC[a-zA-Z0-9_-]{22}$")  # common UC... pattern


def normalize_tags(tags: Iterable[str]) -> List[str]:
    return sorted({re.sub(r"\s+", " ", t).strip().lower() for t in tags if t and t.strip()})

def chunk_text(s: str, max_chars=1000, overlap=150) -> List[str]:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if not s:
        return []
    if len(s) <= max_chars:
        return [s]
    # Ensure we always advance by at least 1 character
    effective_overlap = max(0, min(overlap, max_chars - 1))
    step = max_chars - effective_overlap
    chunks, i = [], 0
    while i < len(s):
        end = min(len(s), i + max_chars)
        chunks.append(s[i:end])
        if end >= len(s):
            break
        i += step
    return chunks

# ----- Rate Limiter for YouTube Transcript API -----
class TranscriptRateLimiter:
    """Rate limiter for YouTube Transcript API: 5 requests per 10 seconds"""
    
    def __init__(self, max_requests=5, time_window=10):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to comply with rate limits (non-recursive, monotonic clock)."""
        while True:
            with self.lock:
                now = time.perf_counter()
                # Drop timestamps outside the window
                self.requests = [t for t in self.requests if now - t < self.time_window]

                if len(self.requests) < self.max_requests:
                    # Record and proceed
                    self.requests.append(now)
                    return

                # Need to wait until the oldest request falls out of the window
                oldest = min(self.requests)
                wait_time = max(0.0, self.time_window - (now - oldest))

            # Release lock while sleeping
            if wait_time > 0:
                print(f"[rate_limit] waiting {wait_time:.2f}s to comply with transcript API limits")
                time.sleep(wait_time)
            else:
                # Loop will re-check immediately
                pass

# Global rate limiter instance
_transcript_rate_limiter = TranscriptRateLimiter()

# ----- YoutubeTranscriptApi -----
def _get_ytt_api():
    return YouTubeTranscriptApi()


def get_transcript_text(video_id: str) -> str:
    try:
        print(f"[transcript] fetching for {video_id}")
        
        # Wait if necessary to comply with rate limits
        _transcript_rate_limiter.wait_if_needed()
        
        ytt_api = _get_ytt_api()
        transcripts = ytt_api.list(video_id)
        try:
            t = transcripts.find_transcript(['en', 'en-US', 'en-GB'])
        except NoTranscriptFound:
            t = transcripts.find_generated_transcript(['en', 'en-US', 'en-GB'])
        segs = t.fetch().to_raw_data()
        text = " ".join(s.get("text", "").strip() for s in segs if s.get("text"))
        print(f"[transcript] ok length={len(text)} for {video_id}")
        return text
    except (TranscriptsDisabled, NoTranscriptFound):
        print(f"[transcript] unavailable for {video_id}")
        return ""
    except Exception:
        print(f"[transcript] error for {video_id}; returning empty")
        return ""

# ----- TagCache -----
CACHE_PATH = os.getenv("TAG_CACHE_PATH", "tag_cache.json")

def _load_cache():
    if CACHE_PATH and os.path.exists(CACHE_PATH):
        try: 
            with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception: 
            return {}
    return {}

def _save_cache(d):
    try: 
        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(d, f)
    except Exception: 
        pass

_yt_client = None
def _get_yt_client():
    global _yt_client
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    if not youtube_api_key: 
        raise ValueError("YOUTUBE_API_KEY environment variable not set")
    if not _yt_client:
        _yt_client = build("youtube", "v3", developerKey=youtube_api_key, static_discovery=False, 
                            cache_discovery=False)
    return _yt_client

def get_uploads_playlist_id(channel_id: str) -> str:
    """Return the playlist ID that contains all uploads for the channel."""
    yt_client = _get_yt_client()
    resp = yt_client.channels().list(part="contentDetails", id=channel_id).execute()
    items = resp.get("items", [])
    if not items:
        return ""
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

def list_channel_uploads(channel_id: str, max_results: int = 25) -> list[str]:
    """List recent uploads using playlistItems (very low quota)."""
    uploads_pl = get_uploads_playlist_id(channel_id)
    if not uploads_pl:
        return []
    yt_client = _get_yt_client()
    req = yt_client.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_pl,
            maxResults=min(max_results, 50),
        )
    resp = req.execute()
    vids = []
    for item in resp.get("items", []):
        vid = (item.get("contentDetails") or {}).get("videoId")
        if vid:
            vids.append(vid)
    return vids

def get_video_meta(video_id: str) -> VideoMeta:
    print(f"[yt] get_video_meta video_id={video_id}")
    yt_client = _get_yt_client()
    resp = yt_client.videos().list(part="snippet", id=video_id).execute()
    items = resp.get("items", [])
    if not items:
        raise ValueError(f"Video not found: {video_id}")
    snippet = items[0]["snippet"]
    tags = normalize_tags(snippet.get("tags", []))
    vm = VideoMeta(
        video_id=video_id,
        title=snippet.get("title", ""),
        channel=snippet.get("channelTitle"),
        channel_id=str(snippet.get("channelId") or ""),
        tags=tags,
        url=YOUTUBE_WATCH_URL_TEMPLATE.format(video_id=video_id),
    )
    print(f"[yt] title='{vm.title}' tags={len(vm.tags)} channel_id={vm.channel_id}")
    return vm

def search_videos_by_tags(tags: list[str], max_per_tag: int = 5, max_search_calls: int = 2) -> list[str]:
    print(f"[search] search_videos_by_tags tags={len(tags or [])} max_per_tag={max_per_tag}")
    ids, calls = set(), 0
    cache = _load_cache()
    today = time.strftime("%Y-%m-%d")
    tags = tags or []
    if not tags:
        return []

    for t in (tags or [])[:3]:
        key = f"{today}:{t}:{max_per_tag}"
        if key in cache:
            print(f"[search] cache hit for tag='{t}' -> {len(cache[key])} ids")
            ids.update(cache[key])
            continue
        if calls >= max_search_calls:
            print("[search] reached max_search_calls; skipping further API calls")
            continue
        try:
            yt_client = _get_yt_client()
            resp = yt_client.search().list(
                part="snippet",
                type="video",
                q=t,
                maxResults=max_per_tag,
                order="relevance",
            ).execute()
            calls += 1
            vids = []
            for item in resp.get("items", []):
                vid = (item.get("id") or {}).get("videoId")
                if vid:
                    ids.add(vid); vids.append(vid)
            cache[key] = vids
        except HttpError as e:
            if getattr(e, "resp", None) and e.resp.status == 403 and b"quota" in getattr(e, "content", b"").lower():
                print("[search] quota exceeded; stopping tag searches for today")
                break
            else:
                print(f"[search] http error on tag='{t}', continuing")
                continue

    _save_cache(cache)
    print(f"[search] total tag-based ids={len(ids)} api_calls={calls}")
    return list(ids)

def search_same_channel_videos(seed_video_id: str, channel_id: str, max_results: int = 25) -> list[str]:
    channel_id = str(channel_id or "")
    # (You may relax this regex if you see non-UC channel IDs in the wild)
    if not CHANNEL_ID_RE.match(channel_id):
        raise ValueError(f"Invalid channel_id '{channel_id}'. Expected a UC... YouTube channel ID.")
    print(f"[search] same-channel uploads channel_id={channel_id} max_results={max_results}")
    vids = list_channel_uploads(channel_id, max_results=max_results)
    print(f"[search] same-channel found {len(vids)} uploads")
    return [v for v in vids if v != seed_video_id]

def discover_by_tags_and_channel(seed_video_id: str, seed_tags: list[str],
                                    channel_id: str, per_tag: int = 5, channel_max: int = 25) -> list[str]:
    # tag_hits uses the quota-aware helper above
    print(f"[discover] seed={seed_video_id} per_tag={per_tag} channel_max={channel_max}")
    tag_hits = set(search_videos_by_tags(seed_tags, max_per_tag=per_tag, max_search_calls=2))
    channel_hits = set(search_same_channel_videos(seed_video_id, channel_id, max_results=channel_max))
    merged = list((tag_hits | channel_hits) - {seed_video_id})
    print(f"[discover] merged candidates={len(merged)} (tags={len(tag_hits)} channel={len(channel_hits)})")
    return merged
        