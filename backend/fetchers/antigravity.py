import base64
import hashlib
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json"
DEFAULT_REDIRECT_URI = "http://localhost:51121/oauth-callback"

OAUTH_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
)

DAILY_CLOUDCODE_BASE = "https://daily-cloudcode-pa.googleapis.com"
PROD_CLOUDCODE_BASE = "https://cloudcode-pa.googleapis.com"
QUOTA_SUMMARY_PATH = "/v1internal:retrieveUserQuotaSummary"
LOAD_CODE_ASSIST_PATH = "/v1internal:loadCodeAssist"
FETCH_MODELS_PATH = "/v1internal:fetchAvailableModels"

DEFAULT_CONSUMER_PROJECT = "aicode-consumers"
# Cloud Code's daily endpoint gates Gemini and Argon buckets on a modern
# Antigravity hub version and changelist in User-Agent.
DEFAULT_USER_AGENT = "antigravity/hub/2.24.0 (aidev_client; os_type=linux; arch=amd64; cl=1004422050)"
API_CLIENT_HEADER = "google-cloud-sdk vscode_cloudshelleditor/0.1"

TOKEN_REFRESH_BUFFER_SECONDS = 300
OAUTH_FLOW_TTL_SECONDS = 900
MAX_PENDING_FLOWS = 50

_pending_oauth_flows: dict[str, dict] = {}


def _env_client_id() -> str:
    return os.getenv("ANTIGRAVITY_OAUTH_CLIENT_ID", "").strip()


def _env_client_secret() -> str:
    return os.getenv("ANTIGRAVITY_OAUTH_CLIENT_SECRET", "").strip()


def _user_agent() -> str:
    return os.getenv("ANTIGRAVITY_USER_AGENT", "").strip() or DEFAULT_USER_AGENT


def _resolve_oauth_client(client_id: str | None = None, client_secret: str | None = None) -> tuple[str, str]:
    cid = (client_id or "").strip() or _env_client_id()
    csec = (client_secret or "").strip() or _env_client_secret()
    return cid, csec


def get_oauth_server_config() -> dict:
    return {
        "has_server_client_id": bool(_env_client_id()),
        "has_server_client_secret": bool(_env_client_secret()),
        "redirect_uri": os.getenv("ANTIGRAVITY_OAUTH_REDIRECT_URI", "").strip() or DEFAULT_REDIRECT_URI,
    }


def _prune_pending_flows(now: float | None = None) -> None:
    now = now if now is not None else time.monotonic()
    expired = [
        state for state, flow in _pending_oauth_flows.items()
        if now - flow["created_at"] > OAUTH_FLOW_TTL_SECONDS
    ]
    for state in expired:
        _pending_oauth_flows.pop(state, None)
    if len(_pending_oauth_flows) > MAX_PENDING_FLOWS:
        oldest = sorted(_pending_oauth_flows.items(), key=lambda kv: kv[1]["created_at"])
        for state, _ in oldest[: len(_pending_oauth_flows) - MAX_PENDING_FLOWS]:
            _pending_oauth_flows.pop(state, None)


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def start_oauth_flow(
    client_id: str | None = None,
    client_secret: str | None = None,
    redirect_uri: str | None = None,
) -> dict:
    cid, csec = _resolve_oauth_client(client_id, client_secret)
    if not cid:
        raise ValueError(
            "OAuth Client ID is required. Set ANTIGRAVITY_OAUTH_CLIENT_ID on the server "
            "or enter your OAuth Client ID below."
        )

    uri = (redirect_uri or "").strip() or get_oauth_server_config()["redirect_uri"]
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(24)

    _prune_pending_flows()
    _pending_oauth_flows[state] = {
        "verifier": verifier,
        "client_id": cid,
        "client_secret": csec,
        "custom_client_id": (client_id or "").strip(),
        "custom_client_secret": (client_secret or "").strip(),
        "redirect_uri": uri,
        "created_at": time.monotonic(),
    }

    params = {
        "client_id": cid,
        "response_type": "code",
        "redirect_uri": uri,
        "scope": " ".join(OAUTH_SCOPES),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    return {
        "status": "ok",
        "auth_url": f"{GOOGLE_AUTH_URL}?{urlencode(params)}",
        "state": state,
        "redirect_uri": uri,
    }


def parse_oauth_callback_input(raw: str) -> tuple[str, str | None]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Paste the redirected URL or authorization code from your browser.")

    query = ""
    if "://" in text:
        parsed = urlparse(text)
        query = parsed.query or parsed.fragment
    elif "code=" in text or "error=" in text:
        query = text.lstrip("?#")

    if query:
        params = parse_qs(query, keep_blank_values=False)
        err = (params.get("error") or [None])[0]
        if err:
            desc = (params.get("error_description") or [None])[0]
            raise ValueError(f"Google OAuth error: {err}" + (f" ({desc})" if desc else ""))
        code = (params.get("code") or [None])[0]
        state = (params.get("state") or [None])[0]
        if not code:
            raise ValueError("No 'code' parameter found in the pasted redirect URL.")
        return code.strip(), state.strip() if state else None

    if any(ch.isspace() for ch in text):
        raise ValueError("Invalid redirect URL or authorization code.")
    return text, None


def _parse_expiry_timestamp(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        return ts if ts > 0 else None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            ts = float(raw)
            if ts > 1e12:
                ts /= 1000.0
            return ts if ts > 0 else None
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    return None


def _format_expiry_iso(expires_in: float | int | None) -> str | None:
    if not isinstance(expires_in, (int, float)) or isinstance(expires_in, bool) or expires_in <= 0:
        return None
    dt = datetime.now(timezone.utc) + timedelta(seconds=float(expires_in))
    return dt.isoformat().replace("+00:00", "Z")


def _extract_first_str(obj: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _unwrap_credential_dict(data: object) -> dict | None:
    if isinstance(data, list):
        for item in data:
            found = _unwrap_credential_dict(item)
            if found:
                return found
        return None
    if not isinstance(data, dict):
        return None

    token_keys = ("refresh_token", "refreshToken", "refresh", "access_token", "accessToken", "access")
    if any(isinstance(data.get(k), str) and data.get(k).strip() for k in token_keys):
        return data

    # Nested containers used by common Antigravity / Cloud Code tools.
    for key in ("google-antigravity", "antigravity", "credentials", "tokens", "oauth", "auth"):
        sub = data.get(key)
        found = _unwrap_credential_dict(sub)
        if found:
            return found

    accounts = data.get("accounts")
    if isinstance(accounts, list) and accounts:
        active_idx = data.get("activeIndex")
        if isinstance(active_idx, int) and 0 <= active_idx < len(accounts):
            found = _unwrap_credential_dict(accounts[active_idx])
            if found:
                return found
        return _unwrap_credential_dict(accounts)

    return None


def parse_antigravity_credentials(raw: str) -> dict:
    """Parse either a raw refresh/access token or JSON exported from Antigravity/ACP tools."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Paste a refresh token or credential JSON.")

    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e

        entry = _unwrap_credential_dict(parsed)
        if not entry:
            raise ValueError("No refresh_token or access_token found in the pasted JSON.")

        refresh_token = _extract_first_str(entry, ("refresh_token", "refreshToken", "refresh"))
        access_token = _extract_first_str(entry, ("access_token", "accessToken", "access"))
        client_id = _extract_first_str(entry, ("client_id", "clientId"))
        client_secret = _extract_first_str(entry, ("client_secret", "clientSecret"))
        project_id = _extract_first_str(
            entry, ("project_id", "projectId", "cloudaicompanionProject", "managedProjectId")
        )
        email = _extract_first_str(entry, ("email", "account_email", "accountEmail"))

        expiry_ts = None
        for key in ("expires_at", "expiresAt", "expiry_date", "expiryDate", "expires"):
            expiry_ts = _parse_expiry_timestamp(entry.get(key))
            if expiry_ts is not None:
                break
        expires_at = (
            datetime.fromtimestamp(expiry_ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            if expiry_ts is not None
            else ""
        )

        if not refresh_token and not access_token:
            raise ValueError("No refresh_token or access_token found in the pasted JSON.")

        return {
            "refresh_token": refresh_token,
            "access_token": access_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "project_id": project_id,
            "email": email,
            "expires_at": expires_at,
        }

    # Raw token pasted directly (strip optional quotes or Bearer prefix).
    cleaned = text.strip().strip('"').strip("'")
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip()
    if any(ch.isspace() for ch in cleaned):
        raise ValueError("Could not recognize token format. Paste a refresh token (1//...) or credential JSON.")

    if cleaned.startswith("ya29."):
        return {
            "refresh_token": "",
            "access_token": cleaned,
            "client_id": "",
            "client_secret": "",
            "project_id": "",
            "email": "",
            "expires_at": "",
        }
    return {
        "refresh_token": cleaned,
        "access_token": "",
        "client_id": "",
        "client_secret": "",
        "project_id": "",
        "email": "",
        "expires_at": "",
    }


def _cloudcode_headers(access_token: str) -> dict:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": _user_agent(),
        "X-Goog-Api-Client": API_CLIENT_HEADER,
    }


def _extract_google_error(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            status_str = err.get("status") or ""
            msg = err.get("message") or ""
            if status_str and msg:
                return f"{status_str}: {msg}"
            if msg:
                return msg
        elif isinstance(err, str):
            desc = body.get("error_description")
            return f"{err}: {desc}" if desc else err
    text = (resp.text or "").strip()
    return text[:200] if text else f"HTTP {resp.status_code}"


async def _fetch_userinfo_email(client: httpx.AsyncClient, access_token: str) -> str | None:
    try:
        r = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if r.status_code == 200:
            email = r.json().get("email")
            if isinstance(email, str) and email.strip():
                return email.strip()
    except Exception as e:
        logger.debug("Antigravity userinfo lookup skipped: %s", e)
    return None


def _extract_project_and_plan(payload: dict) -> tuple[str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    project = payload.get("cloudaicompanionProject")
    project_id = None
    if isinstance(project, str) and project.strip():
        project_id = project.strip()
    elif isinstance(project, dict):
        pid = project.get("id") or project.get("projectId")
        if isinstance(pid, str) and pid.strip():
            project_id = pid.strip()

    plan = None
    for candidate in (
        (payload.get("paidTier") or {}).get("name") if isinstance(payload.get("paidTier"), dict) else None,
        (payload.get("currentTier") or {}).get("name") if isinstance(payload.get("currentTier"), dict) else None,
        payload.get("tierDisplayName"),
        (payload.get("planInfo") or {}).get("planName") if isinstance(payload.get("planInfo"), dict) else None,
        (payload.get("currentTier") or {}).get("id") if isinstance(payload.get("currentTier"), dict) else None,
    ):
        if isinstance(candidate, str) and candidate.strip():
            plan = candidate.strip()
            break

    return project_id, plan


async def _load_code_assist(client: httpx.AsyncClient, access_token: str) -> tuple[str | None, str | None]:
    body = {
        "metadata": {
            "ideType": "ANTIGRAVITY",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
    }
    headers = _cloudcode_headers(access_token)
    for base in (DAILY_CLOUDCODE_BASE, PROD_CLOUDCODE_BASE):
        try:
            r = await client.post(
                f"{base}{LOAD_CODE_ASSIST_PATH}",
                headers=headers,
                json=body,
                timeout=12,
            )
            if r.status_code == 200:
                project_id, plan = _extract_project_and_plan(r.json())
                if project_id or plan:
                    return project_id, plan
        except Exception as e:
            logger.debug("Antigravity loadCodeAssist failed on %s: %s", base, e)
    return None, None


async def exchange_oauth_code(
    callback_input: str,
    state: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    redirect_uri: str | None = None,
) -> dict:
    code, url_state = parse_oauth_callback_input(callback_input)
    _prune_pending_flows()

    if url_state and state and url_state != state:
        raise ValueError("OAuth state mismatch. Please generate a new sign-in link and try again.")

    lookup_state = url_state or (state or "").strip()
    flow = _pending_oauth_flows.pop(lookup_state, None) if lookup_state else None

    if flow is not None:
        cid = flow["client_id"]
        csec = flow["client_secret"]
        uri = flow["redirect_uri"]
        verifier = flow["verifier"]
        custom_cid = flow["custom_client_id"]
        custom_csec = flow["custom_client_secret"]
    else:
        cid, csec = _resolve_oauth_client(client_id, client_secret)
        if not cid:
            raise ValueError("OAuth session expired or Client ID missing. Generate a new Google sign-in link.")
        uri = (redirect_uri or "").strip() or get_oauth_server_config()["redirect_uri"]
        verifier = None
        custom_cid = (client_id or "").strip()
        custom_csec = (client_secret or "").strip()

    form = {
        "client_id": cid,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": uri,
    }
    if csec:
        form["client_secret"] = csec
    if verifier:
        form["code_verifier"] = verifier

    async with httpx.AsyncClient() as client:
        r = await client.post(GOOGLE_TOKEN_URL, data=form, timeout=15)
        if r.status_code != 200:
            raise ValueError(f"Token exchange failed: {_extract_google_error(r)}")
        tok = r.json()
        access_token = tok.get("access_token") or ""
        refresh_token = tok.get("refresh_token") or ""
        expires_at = _format_expiry_iso(tok.get("expires_in"))

        if not access_token and not refresh_token:
            raise ValueError("Google returned no access_token or refresh_token.")

        email = await _fetch_userinfo_email(client, access_token) if access_token else None
        project_id, plan = await _load_code_assist(client, access_token) if access_token else (None, None)

    return {
        "status": "ok",
        "refresh_token": refresh_token,
        "access_token": access_token,
        "expires_at": expires_at or "",
        "email": email or "",
        "project_id": project_id or "",
        "plan": plan or "",
        "client_id": custom_cid,
        "client_secret": custom_csec,
    }


async def _refresh_access_token(
    client: httpx.AsyncClient,
    refresh_token: str,
    client_id: str,
    client_secret: str,
) -> tuple[str, str | None, str | None]:
    if not client_id:
        raise ValueError(
            "OAuth Client ID is required to refresh Antigravity tokens. "
            "Configure ANTIGRAVITY_OAUTH_CLIENT_ID on the server or fill OAuth Client ID in Settings."
        )
    form = {
        "client_id": client_id,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    if client_secret:
        form["client_secret"] = client_secret

    r = await client.post(GOOGLE_TOKEN_URL, data=form, timeout=15)
    if r.status_code != 200:
        detail = _extract_google_error(r)
        if r.status_code in (400, 401):
            raise PermissionError(f"OAuth token refresh rejected ({detail}) — reconnect Antigravity in Settings")
        raise RuntimeError(f"OAuth token refresh failed (HTTP {r.status_code}): {detail}")

    tok = r.json()
    access_token = tok.get("access_token")
    if not access_token:
        raise RuntimeError("OAuth token refresh response did not contain an access_token")
    expires_at = _format_expiry_iso(tok.get("expires_in"))
    rotated_refresh = tok.get("refresh_token") if isinstance(tok.get("refresh_token"), str) else None
    return access_token, expires_at, rotated_refresh


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return slug or "quota"


def _canonical_group_id(display_name: str, bucket_ids: list[str]) -> str:
    name_l = (display_name or "").strip().lower()
    joined_ids = " ".join(b.lower() for b in bucket_ids)
    if "argon" in name_l or "argon" in joined_ids:
        return "argon"
    if "claude" in name_l or "gpt" in name_l or any(b.lower().startswith("3p") for b in bucket_ids):
        return "claude_gpt"
    if "gemini" in name_l or any(b.lower().startswith("gemini") for b in bucket_ids):
        return "gemini"
    return _slugify(display_name)


def _short_group_label(group_id: str, display_name: str) -> str:
    if group_id == "gemini":
        return "Gemini"
    if group_id == "claude_gpt":
        return "Claude & GPT"
    if group_id == "argon":
        return "Argon"
    return (display_name or group_id).strip()


def _normalize_window(window: str | None, bucket_id: str = "") -> str:
    raw = (window or "").strip().lower()
    bid = (bucket_id or "").strip().lower()
    if raw in ("5h", "5_hour", "5-hour", "session", "five_hour") or bid.endswith("-5h") or "5h" in bid:
        return "5h"
    if raw in ("weekly", "week", "7d", "seven_day") or "weekly" in bid or "week" in bid:
        return "weekly"
    return raw or "quota"


def _window_label(window: str) -> str:
    if window == "5h":
        return "5-hour"
    if window == "weekly":
        return "Weekly"
    return window.replace("_", " ").capitalize()


def _parse_remaining_fraction(bucket: dict) -> float | None:
    if bucket.get("isExhausted") is True:
        return 0.0
    raw = bucket.get("remainingFraction")
    if raw is None:
        # Proto3 omits default 0.0 float fields on the wire when a bucket is exhausted.
        if bucket.get("resetTime"):
            return 0.0
        return None
    if isinstance(raw, bool):
        return None
    try:
        frac = float(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, frac))


def _normalize_summary_payload(payload: dict) -> list[dict]:
    """Parse retrieveUserQuotaSummary into canonical groups[]."""
    if not isinstance(payload, dict):
        return []

    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list):
        raw_buckets = payload.get("buckets")
        if isinstance(raw_buckets, list) and raw_buckets:
            raw_groups = [{"displayName": "Gemini Models", "buckets": raw_buckets}]
        else:
            return []

    groups_by_id: dict[str, dict] = {}
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            continue
        raw_buckets = raw_group.get("buckets")
        if not isinstance(raw_buckets, list):
            continue
        display_name = str(raw_group.get("displayName") or "").strip()
        bucket_ids = [str(b.get("bucketId") or "") for b in raw_buckets if isinstance(b, dict)]
        group_id = _canonical_group_id(display_name, bucket_ids)
        group_name = display_name or _short_group_label(group_id, group_id)

        parsed_buckets = []
        for b in raw_buckets:
            if not isinstance(b, dict):
                continue
            rem_frac = _parse_remaining_fraction(b)
            if rem_frac is None:
                continue
            bucket_id = str(b.get("bucketId") or "").strip()
            window = _normalize_window(b.get("window"), bucket_id)
            if not bucket_id:
                bucket_id = f"{group_id}-{window}"
            pct_remaining = round(rem_frac * 100.0, 1)
            pct_used = round((1.0 - rem_frac) * 100.0, 1)
            # Untouched buckets report a synthetic rolling resetTime = now + window.
            # Hide resets_at when 100% remaining so dashboards don't show a frozen countdown.
            raw_reset = b.get("resetTime")
            resets_at = raw_reset.strip() if isinstance(raw_reset, str) and raw_reset.strip() and rem_frac < 1.0 else None
            parsed_buckets.append({
                "id": bucket_id,
                "window": window,
                "label": _window_label(window),
                "percent_used": pct_used,
                "percent_remaining": pct_remaining,
                "resets_at": resets_at,
            })

        if not parsed_buckets:
            continue

        if group_id in groups_by_id:
            existing_ids = {b["id"] for b in groups_by_id[group_id]["buckets"]}
            for b in parsed_buckets:
                if b["id"] not in existing_ids:
                    groups_by_id[group_id]["buckets"].append(b)
                    existing_ids.add(b["id"])
        else:
            groups_by_id[group_id] = {
                "id": group_id,
                "name": group_name,
                "buckets": parsed_buckets,
            }

    return _sort_groups(list(groups_by_id.values()))


def _merge_missing_groups(primary: list[dict], fallback: list[dict]) -> list[dict]:
    """Merge buckets from fallback only when their group/bucket is absent from primary."""
    by_id = {g["id"]: {**g, "buckets": list(g["buckets"])} for g in primary}
    for fg in fallback:
        gid = fg["id"]
        if gid not in by_id:
            by_id[gid] = {**fg, "buckets": list(fg["buckets"])}
            continue
        existing = {b["id"] for b in by_id[gid]["buckets"]}
        for b in fg["buckets"]:
            if b["id"] not in existing:
                by_id[gid]["buckets"].append(b)
                existing.add(b["id"])
    return _sort_groups(list(by_id.values()))


def _sort_groups(groups: list[dict]) -> list[dict]:
    group_rank = {"gemini": 0, "claude_gpt": 1, "argon": 2}
    window_rank = {"5h": 0, "weekly": 1}

    out = []
    for g in sorted(groups, key=lambda item: (group_rank.get(item["id"], 99), item["name"].lower())):
        buckets = sorted(
            g["buckets"],
            key=lambda b: (window_rank.get(b["window"], 99), b["id"]),
        )
        out.append({**g, "buckets": buckets})
    return out


def _normalize_models_fallback(payload: dict) -> list[dict]:
    """Fallback parser for /v1internal:fetchAvailableModels when summary is empty."""
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, dict):
        return []

    buckets_by_group: dict[str, tuple[str, str, float, str | None]] = {}
    for model_id, info in models.items():
        if not isinstance(info, dict):
            continue
        qinfo = info.get("quotaInfo")
        if not isinstance(qinfo, dict):
            continue
        rem_frac = _parse_remaining_fraction(qinfo)
        if rem_frac is None:
            continue
        mid = str(model_id).lower()
        if "argon" in mid:
            gid, gname, window = "argon", "Gemini 4 Argon", "weekly"
        elif "claude" in mid or "gpt" in mid:
            gid, gname, window = "claude_gpt", "Claude and GPT models", "5h"
        elif "gemini" in mid:
            gid, gname, window = "gemini", "Gemini Models", "5h"
        else:
            continue

        raw_reset = qinfo.get("resetTime")
        resets_at = raw_reset.strip() if isinstance(raw_reset, str) and raw_reset.strip() and rem_frac < 1.0 else None
        prev = buckets_by_group.get(gid)
        if prev is None or rem_frac < prev[2]:
            buckets_by_group[gid] = (gname, window, rem_frac, resets_at)

    groups = []
    for gid, (gname, window, rem_frac, resets_at) in buckets_by_group.items():
        groups.append({
            "id": gid,
            "name": gname,
            "buckets": [{
                "id": f"{gid}-{window}",
                "window": window,
                "label": _window_label(window),
                "percent_used": round((1.0 - rem_frac) * 100.0, 1),
                "percent_remaining": round(rem_frac * 100.0, 1),
                "resets_at": resets_at,
            }],
        })
    return _sort_groups(groups)


def _flatten_limits(groups: list[dict]) -> list[dict]:
    limits = []
    for g in groups:
        short = _short_group_label(g["id"], g["name"])
        for b in g["buckets"]:
            win_tag = "5h" if b["window"] == "5h" else b["label"]
            limits.append({
                "id": b["id"],
                "group": g["id"],
                "group_name": g["name"],
                "window": b["window"],
                "label": f"{short} · {win_tag}",
                "percent_used": b["percent_used"],
                "percent_remaining": b["percent_remaining"],
                "resets_at": b["resets_at"],
            })
    return limits


async def _query_quota(
    client: httpx.AsyncClient,
    access_token: str,
    project_id: str,
) -> tuple[list[dict], int | None, str | None]:
    """Query daily then prod retrieveUserQuotaSummary, falling back to fetchAvailableModels."""
    headers = _cloudcode_headers(access_token)
    body = {"project": project_id}
    daily_groups: list[dict] = []
    last_status: int | None = None
    last_error: str | None = None

    for base in (DAILY_CLOUDCODE_BASE, PROD_CLOUDCODE_BASE):
        try:
            r = await client.post(
                f"{base}{QUOTA_SUMMARY_PATH}",
                headers=headers,
                json=body,
                timeout=15,
            )
            last_status = r.status_code
            if r.status_code == 401:
                return [], 401, _extract_google_error(r)
            if r.status_code != 200:
                last_error = f"HTTP {r.status_code}: {_extract_google_error(r)}"
                continue
            groups = _normalize_summary_payload(r.json())
            if base == DAILY_CLOUDCODE_BASE:
                daily_groups = groups
                has_gemini = any(g["id"] == "gemini" for g in daily_groups)
                if daily_groups and has_gemini:
                    return daily_groups, 200, None
                # If daily succeeded with partial groups (missing Gemini) or empty,
                # check prod to fill in any missing group without overwriting daily values.
            else:
                if daily_groups:
                    return _merge_missing_groups(daily_groups, groups), 200, None
                if groups:
                    return groups, 200, None
        except httpx.HTTPError as e:
            last_error = str(e)

    if daily_groups:
        return daily_groups, 200, None

    # Fallback: per-model quotaInfo from fetchAvailableModels.
    for base in (DAILY_CLOUDCODE_BASE, PROD_CLOUDCODE_BASE):
        try:
            r = await client.post(
                f"{base}{FETCH_MODELS_PATH}",
                headers=headers,
                json=body,
                timeout=15,
            )
            if r.status_code == 401:
                return [], 401, _extract_google_error(r)
            if r.status_code == 200:
                groups = _normalize_models_fallback(r.json())
                if groups:
                    return groups, 200, None
        except httpx.HTTPError as e:
            last_error = last_error or str(e)

    return [], last_status, last_error


async def fetch_antigravity(config: dict = None, on_config_update=None) -> dict:
    if config is None:
        config = {}

    refresh_token = (config.get("antigravity_refresh_token") or "").strip()
    access_token = (config.get("antigravity_access_token") or "").strip()
    expires_ts = _parse_expiry_timestamp(config.get("antigravity_token_expires_at"))
    client_id, client_secret = _resolve_oauth_client(
        config.get("antigravity_client_id"),
        config.get("antigravity_client_secret"),
    )
    configured_project = (config.get("antigravity_project_id") or "").strip()
    configured_email = (config.get("antigravity_email") or "").strip()

    if not refresh_token and not access_token:
        return {"configured": False, "error": "Not configured"}

    updates: dict = {}

    async with httpx.AsyncClient() as client:
        now = time.time()
        needs_refresh = not access_token or (
            expires_ts is not None and expires_ts - now <= TOKEN_REFRESH_BUFFER_SECONDS
        )
        refreshed_this_call = False

        if needs_refresh:
            if not refresh_token:
                return {
                    "configured": True,
                    "status": "error",
                    "error": "Access token expired and no refresh token is configured — reconnect in Settings",
                }
            try:
                access_token, new_expires_at, rotated_refresh = await _refresh_access_token(
                    client, refresh_token, client_id, client_secret
                )
                refreshed_this_call = True
                updates["antigravity_access_token"] = access_token
                if new_expires_at:
                    updates["antigravity_token_expires_at"] = new_expires_at
                if rotated_refresh and rotated_refresh != refresh_token:
                    refresh_token = rotated_refresh
                    updates["antigravity_refresh_token"] = rotated_refresh
            except (ValueError, PermissionError, RuntimeError) as e:
                return {"configured": True, "status": "error", "error": str(e)}
            except Exception as e:
                return {"configured": True, "status": "error", "error": str(e)}

        discovered_project, plan = await _load_code_assist(client, access_token)
        project_id = discovered_project or configured_project or DEFAULT_CONSUMER_PROJECT
        if discovered_project and discovered_project != configured_project:
            updates["antigravity_project_id"] = discovered_project

        groups, status_code, err_detail = await _query_quota(client, access_token, project_id)

        # Retry once after forcing a token refresh if the cached access token was rejected.
        if status_code == 401 and refresh_token and not refreshed_this_call:
            try:
                access_token, new_expires_at, rotated_refresh = await _refresh_access_token(
                    client, refresh_token, client_id, client_secret
                )
                updates["antigravity_access_token"] = access_token
                if new_expires_at:
                    updates["antigravity_token_expires_at"] = new_expires_at
                if rotated_refresh and rotated_refresh != refresh_token:
                    refresh_token = rotated_refresh
                    updates["antigravity_refresh_token"] = rotated_refresh
            except (ValueError, PermissionError, RuntimeError) as e:
                return {"configured": True, "status": "error", "error": str(e)}
            except Exception as e:
                return {"configured": True, "status": "error", "error": str(e)}

            discovered_project, plan = await _load_code_assist(client, access_token)
            project_id = discovered_project or configured_project or DEFAULT_CONSUMER_PROJECT
            if discovered_project and discovered_project != configured_project:
                updates["antigravity_project_id"] = discovered_project
            groups, status_code, err_detail = await _query_quota(client, access_token, project_id)

        email = configured_email
        if not email and groups:
            fetched_email = await _fetch_userinfo_email(client, access_token)
            if fetched_email:
                email = fetched_email
                updates["antigravity_email"] = fetched_email

        if updates and on_config_update:
            try:
                await on_config_update(updates)
            except Exception:
                logger.exception("Failed to persist updated Antigravity token state")

        if status_code == 401:
            return {
                "configured": True,
                "status": "error",
                "error": "Session expired (HTTP 401) — reconnect Antigravity in Settings",
            }
        if not groups:
            return {
                "configured": True,
                "status": "error",
                "error": err_detail or "No Antigravity quota data returned for this account",
            }

        return {
            "configured": True,
            "status": "ok",
            "data": {
                "plan": plan,
                "email": email or None,
                "project_id": project_id,
                "groups": groups,
                "limits": _flatten_limits(groups),
            },
        }
