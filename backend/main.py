import asyncio
import base64
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote

import pyotp
import qrcode
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Header, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Loaded before the local modules below so their own module-level os.getenv()
# calls (oidc.py, email_notifier.py, fetchers/chatgpt.py, ...) see it too.
# Resolved relative to this file rather than cwd, so it's found the same way
# whether uvicorn is started from the repo root or from backend/.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("openusage")

import db
from db import (
    init_db,
    create_user,
    get_user_by_username,
    get_user_by_id,
    create_session,
    get_user_by_session,
    delete_session,
    delete_all_sessions_for_user,
    get_user_services,
    get_all_user_services,
    get_user_service,
    add_user_service,
    update_user_service,
    delete_user_service,
    delete_all_user_services,
    delete_user_account,
    save_service_usage,
    count_service_usage_history,
    get_service_usage_history,
    get_latest_service_usage,
    verify_password,
    get_categories,
    create_category,
    update_category,
    delete_category,
    move_category,
    get_user_by_email,
    update_user_email,
    update_user_password,
    create_oidc_user,
    create_oidc_state,
    consume_oidc_state,
    create_password_reset_token,
    get_password_reset_token,
    delete_password_reset_token,
    get_service_notify_armed,
    set_service_notify_armed,
    add_push_subscription,
    delete_push_subscription,
)
import email_notifier
import oidc
import two_factor as mfa
import push_notifier
from thresholds import APPLICABLE_THRESHOLDS, extract_history_metrics, extract_usage_percentages
from curl_parse import parse_curl, parse_gemini_curl, parse_openai_session_curl, parse_chatgpt_curl, parse_ollama_curl, parse_opencode_curl
from fetchers.claude import fetch_claude
from fetchers.gemini import fetch_gemini
from fetchers.deepseek import fetch_deepseek
from fetchers.copilot import fetch_copilot, start_device_flow, poll_device_flow
from fetchers.openai_api import fetch_openai
from fetchers.mistral import fetch_mistral
from fetchers.chatgpt import fetch_chatgpt
from fetchers.ollama_cloud import fetch_ollama_cloud
from fetchers.opencode import fetch_opencode
from fetchers.generic import fetch_generic, test_generic

ENABLE_ACCOUNT_CREATION = os.getenv("ENABLE_ACCOUNT_CREATION", "true").lower() in ("true", "1", "yes")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
PASSWORD_RESET_MAX_AGE = timedelta(hours=1)
DEFAULT_REFRESH_INTERVAL_MINUTES = 5
REFRESH_SCHEDULER_TICK_SECONDS = 60
# Under healthy operation each service refreshes within its configured window,
# so data older than this multiple can only mean live refreshes have been
# failing -- even across a process restart, where the in-memory trackers below
# start out empty.
STALE_DATA_MULTIPLIER = 3


SERVICE_FETCHERS = {
    "claude": fetch_claude,
    "gemini": fetch_gemini,
    "deepseek": fetch_deepseek,
    "copilot": fetch_copilot,
    "openai": fetch_openai,
    "mistral": fetch_mistral,
    "chatgpt": fetch_chatgpt,
    "ollama_cloud": fetch_ollama_cloud,
    "opencode": fetch_opencode,
    "custom": fetch_generic,
}

# Tracks the most recent live-fetch failure per service, so requests that hit
# the fast cached-data path (below) can still surface that the background
# refresh loop is failing, not just requests that trigger a live fetch themselves.
_last_fetch_error: dict[int, str] = {}

# Failed attempts are not written to service_usage_history, but they still need
# to respect the service's configured cadence. Without this timestamp, a failed
# endpoint with stale cached data would be retried every scheduler tick.
_last_fetch_attempt_at: dict[int, datetime] = {}

# Coalesces concurrent live-fetch requests for the same service into one
# in-flight task. Without this, a foreground force-refresh (e.g. the
# dashboard's "Refresh" button, or a poll re-triggered because the service is
# already erroring) can overlap with the background refresh loop's own pass
# for the same service. Each caller fetches its own independent copy of
# `user_service["config"]` from the DB, so two concurrent Gemini fetches each
# rotate the session cookie against their own stale snapshot and then race to
# persist it -- whichever write lands last silently discards the other's
# newer (still-valid) rotated cookie, so the next fetch replays an
# already-invalidated cookie and the session looks "expired" again. Routing
# every caller through the same Task removes the race entirely.
_inflight_fetches: dict[int, asyncio.Task] = {}


def _clear_service_runtime_state(service_ids: list[int]) -> None:
    for service_id in service_ids:
        _last_fetch_error.pop(service_id, None)
        _last_fetch_attempt_at.pop(service_id, None)


def _get_or_start_live_fetch(user_service: dict) -> asyncio.Task:
    svc_id = user_service["id"]
    task = _inflight_fetches.get(svc_id)
    if task is None or task.done():
        task = asyncio.create_task(_live_fetch_and_save(user_service))
        _inflight_fetches[svc_id] = task

        def _cleanup(_task: asyncio.Task, svc_id: int = svc_id) -> None:
            if _inflight_fetches.get(svc_id) is _task:
                _inflight_fetches.pop(svc_id, None)

        task.add_done_callback(_cleanup)
    return task


async def _check_thresholds_and_notify(user_service: dict, data: dict):
    """Fires a push notification the moment usage crosses a configured
    threshold, once per crossing -- see service_notify_state's "armed" flag
    in db.py for how the moment usage drops back below it re-arms the check.
    """
    svc_id = user_service["id"]
    svc_type = user_service["service_type"]
    thresholds_cfg = user_service.get("notify_thresholds") or {}
    if not thresholds_cfg:
        return

    percentages = extract_usage_percentages(svc_type, data)
    if not percentages:
        return

    for threshold_type in APPLICABLE_THRESHOLDS.get(svc_type, ()):
        threshold = thresholds_cfg.get(f"{threshold_type}_threshold")
        pct = percentages.get(threshold_type)
        if threshold is None or pct is None:
            continue

        armed = await get_service_notify_armed(svc_id, threshold_type)
        if pct >= threshold:
            if armed:
                try:
                    await push_notifier.send_push(
                        user_service["user_id"],
                        f"{user_service['name']}：{dict(session='会话额度', weekly='每周额度', monthly='每月额度').get(threshold_type, threshold_type)}已用 {pct:.0f}%",
                        f"用量已超过设置的 {threshold:.0f}% 阈值。",
                        url="/",
                    )
                except push_notifier.PushNotifierError as e:
                    logger.info("Skipped threshold push for service id=%s: %s", svc_id, e)
                else:
                    # Only a successful delivery consumes this crossing. If
                    # every device failed (or none are subscribed), leave it
                    # armed so a later refresh can retry.
                    await set_service_notify_armed(svc_id, threshold_type, False)
        elif not armed:
            await set_service_notify_armed(svc_id, threshold_type, True)


async def _live_fetch_and_save(user_service: dict) -> dict:
    svc_id = user_service["id"]
    svc_type = user_service["service_type"]
    svc_name = user_service["name"]
    config = user_service.get("config", {})
    # Config version captured at task start: if the user changes the config
    # while this task is in flight, the task's writeback and result are
    # discarded instead of clobbering the newer settings.
    config_version = user_service.get("config_version", 1)
    _last_fetch_attempt_at[svc_id] = datetime.now(timezone.utc)

    fetcher = SERVICE_FETCHERS.get(svc_type)
    if not fetcher:
        return {
            "id": svc_id,
            "service_type": svc_type,
            "name": svc_name,
            "result": {"configured": False, "error": f"Unknown service type: {svc_type}"},
        }

    async def _on_config_update(updates: dict):
        # Credential rotation etc. writes back the fetcher's config updates.
        # Atomic compare-and-set on config_version: if the user changed the
        # config while this task was in flight, the write matches no row
        # (returns False) and cannot clobber the newer settings.
        nonlocal config_version
        merged = {**config, **updates}
        write_ok = await update_user_service(
            svc_id, user_service["user_id"], svc_name, merged,
            user_service.get("category_id"),
            expected_config_version=config_version,
        )
        if write_ok:
            config.update(updates)
            config_version += 1
        else:
            logger.info(
                "Skipped config writeback for %s (id=%s): config changed during fetch",
                svc_name, svc_id,
            )

    try:
        logger.info("Starting live fetch for %s (%s, id=%s)...", svc_name, svc_type, svc_id)
        if svc_type == "gemini":
            result = await fetcher(config, on_config_update=_on_config_update, service_id=svc_id)
        elif svc_type == "chatgpt":
            result = await fetcher(config, on_config_update=_on_config_update)
        else:
            result = await fetcher(config)

        if result.get("status") == "ok":
            data = result.get("data", {})
            # Atomic version guard at the write point: a config update that
            # landed while this task was in flight (credential swap, rename,
            # category move) makes the insert match no row — the stale fetch's
            # result is discarded, never persisted over the newer settings.
            saved = await save_service_usage(svc_id, data, expected_config_version=config_version)
            if not saved:
                current = await get_user_service(svc_id, user_service["user_id"])
                if current is None:
                    # The service was deleted while its provider request was in
                    # flight. Do not recreate orphaned history or send alerts.
                    _clear_service_runtime_state([svc_id])
                    return {"id": svc_id, "service_type": svc_type, "name": svc_name, "result": result}
                logger.info(
                    "Discarding stale fetch for %s (id=%s): config changed during fetch",
                    svc_name, svc_id,
                )
                # Re-fetch once AFTER this task has fully exited: calling
                # _trigger_background_live_fetch now would just join this very
                # task (it still owns the inflight slot), so the follow-up is
                # deferred until the done-callback clears the slot.
                self_task = asyncio.current_task()

                async def _refetch_after_exit():
                    for _ in range(100):
                        if _inflight_fetches.get(svc_id) is not self_task:
                            _get_or_start_live_fetch(current)
                            return
                        await asyncio.sleep(0.05)

                asyncio.create_task(_refetch_after_exit())
                return {
                    "id": svc_id,
                    "service_type": svc_type,
                    "name": current["name"],
                    "result": {
                        "configured": True,
                        "status": "ok",
                        "data": data,
                        "stale_error": "配置已变更，本次结果已丢弃并按新配置重新刷新",
                        "discarded_config_changed": True,
                    },
                }
            _last_fetch_error.pop(svc_id, None)
            try:
                await _check_thresholds_and_notify(user_service, data)
            except Exception:
                logger.exception("Threshold notify check failed for %s (id=%s)", svc_name, svc_id)
            logger.info("Live fetch succeeded for %s (id=%s)", svc_name, svc_id)
            return {"id": svc_id, "service_type": svc_type, "name": svc_name, "result": result}

        err_msg = result.get("error") or "Live refresh failed"
        _last_fetch_error[svc_id] = err_msg
        logger.warning("Live fetch failed for %s (id=%s): %s", svc_name, svc_id, err_msg)
        cached_data, fetched_at = await get_latest_service_usage(svc_id)
        if cached_data:
            return {
                "id": svc_id,
                "service_type": svc_type,
                "name": svc_name,
                "result": {
                    "configured": True,
                    "status": "ok",
                    "data": cached_data,
                    "cached": True,
                    "fetched_at": fetched_at,
                    "stale_error": err_msg,
                },
            }
        return {"id": svc_id, "service_type": svc_type, "name": svc_name, "result": result}
    except Exception as e:
        _last_fetch_error[svc_id] = str(e)
        logger.exception("Live fetch exception for %s (id=%s): %s", svc_name, svc_id, e)
        cached_data, fetched_at = await get_latest_service_usage(svc_id)
        if cached_data:
            return {
                "id": svc_id,
                "service_type": svc_type,
                "name": svc_name,
                "result": {
                    "configured": True,
                    "status": "ok",
                    "data": cached_data,
                    "cached": True,
                    "fetched_at": fetched_at,
                    "stale_error": str(e),
                },
            }
        return {
            "id": svc_id,
            "service_type": svc_type,
            "name": svc_name,
            "result": {"configured": True, "status": "error", "error": str(e)},
        }


def _trigger_background_live_fetch(user_service: dict):
    # Fire-and-forget: joins whatever fetch is already in flight for this
    # service instead of starting a redundant one.
    _get_or_start_live_fetch(user_service)


async def _fetch_and_save_user_service(
    user_service: dict, force_refresh: bool = False, max_age_seconds: int | None = None
) -> dict:
    svc_id = user_service["id"]
    svc_type = user_service["service_type"]
    svc_name = user_service["name"]
    if max_age_seconds is None:
        try:
            interval_minutes = int(
                user_service.get("refresh_interval_minutes", DEFAULT_REFRESH_INTERVAL_MINUTES)
            )
        except (TypeError, ValueError):
            interval_minutes = DEFAULT_REFRESH_INTERVAL_MINUTES
        max_age_seconds = max(1, interval_minutes) * 60

    # 1. If not force_refresh, check if we have any cached data
    if not force_refresh:
        inflight = _inflight_fetches.get(svc_id)
        if inflight is not None and not inflight.done():
            return await inflight

        last_attempt = _last_fetch_attempt_at.get(svc_id)
        attempt_age = (
            (datetime.now(timezone.utc) - last_attempt).total_seconds()
            if last_attempt is not None
            else None
        )
        attempt_is_due = attempt_age is None or attempt_age >= max_age_seconds
        cached_data, fetched_at = await get_latest_service_usage(svc_id)
        if cached_data and fetched_at:
            age = None
            try:
                fetch_time = datetime.fromisoformat(fetched_at)
                age = (datetime.now(timezone.utc) - fetch_time).total_seconds()
            except Exception:
                pass

            stale_error = _last_fetch_error.get(svc_id)
            if not stale_error and age is not None and age >= max_age_seconds * STALE_DATA_MULTIPLIER:
                stale_error = f"No successful refresh since {fetched_at}"

            if age is not None and age >= max_age_seconds and attempt_is_due:
                _trigger_background_live_fetch(user_service)

            return {
                "id": svc_id,
                "service_type": svc_type,
                "name": svc_name,
                "result": {
                    "configured": True,
                    "status": "ok",
                    "data": cached_data,
                    "cached": True,
                    "fetched_at": fetched_at,
                    "stale_error": stale_error,
                },
            }

        if not attempt_is_due and svc_id in _last_fetch_error:
            return {
                "id": svc_id,
                "service_type": svc_type,
                "name": svc_name,
                "result": {
                    "configured": True,
                    "status": "error",
                    "error": _last_fetch_error[svc_id],
                },
            }

    # 2. Force refresh or no cached data yet -> live fetch (joins an
    # in-flight fetch for this service if one is already running)
    return await _get_or_start_live_fetch(user_service)


async def _refresh_loop():
    await asyncio.sleep(5)
    while True:
        try:
            all_services = await get_all_user_services()
            if all_services:
                await asyncio.gather(
                    *(_fetch_and_save_user_service(svc) for svc in all_services)
                )
        except Exception:
            pass
        await asyncio.sleep(REFRESH_SCHEDULER_TICK_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    task = asyncio.create_task(_refresh_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Authentication dependency
async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="缺少认证请求头"
        )
    token = authorization.replace("Bearer ", "").strip()
    user = await get_user_by_session(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="登录会话无效或已过期"
        )
    return user


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/version")
async def version():
    # build = 部署构建标识（compose 的 OPENUSAGE_BUILD_ID，如 "zh-20261121g"），
    # 用于核对运行容器与本地真源的对应关系；version 字段保留供旧客户端读取。
    return {
        "version": os.getenv("APP_VERSION", "dev"),
        "build": os.getenv("OPENUSAGE_BUILD_ID", "unknown"),
    }


@app.get("/api/auth/config")
async def auth_config():
    return {"enable_account_creation": ENABLE_ACCOUNT_CREATION}


def _public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user.get("email"),
        "is_admin": bool(user.get("is_admin")),
    }


def _normalize_email(email: Optional[str]) -> Optional[str]:
    if not email:
        return None
    email = email.strip().lower()
    if "@" not in email or " " in email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请输入有效邮箱地址")
    return email


class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None


class AuthRequest(BaseModel):
    username: str
    password: str


class TwoFactorVerifyRequest(BaseModel):
    challenge_token: str
    code: str


@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    if not ENABLE_ACCOUNT_CREATION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="账户注册已关闭"
        )
    username = req.username.strip().lower()
    if len(username) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="用户名至少需要 3 个字符"
        )
    if len(req.password) < 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="密码至少需要 4 个字符"
        )
    email = _normalize_email(req.email)

    existing = await get_user_by_username(username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="用户名已被使用"
        )
    if email and await get_user_by_email(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="邮箱已注册"
        )

    user_id = await create_user(username, req.password, email)
    token = await create_session(user_id)
    user = await get_user_by_username(username)
    return {"token": token, "user": _public_user(user)}


@app.post("/api/auth/login")
async def login(req: AuthRequest):
    username = req.username.strip().lower()
    user = await get_user_by_username(username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误"
        )

    challenge_token = await mfa.begin_login(user["id"])
    if challenge_token:
        return {"requires_2fa": True, "challenge_token": challenge_token}

    token = await create_session(user["id"])
    return {"token": token, "user": _public_user(user)}


@app.post("/api/auth/2fa/verify")
async def two_factor_verify(req: TwoFactorVerifyRequest):
    row = await db.get_2fa_by_challenge_hash(mfa.digest(req.challenge_token))
    if not row or not row["enabled"]:
        raise HTTPException(status_code=401, detail="请重新登录以验证验证码")
    if not row["challenge_expires_at"] or datetime.fromisoformat(row["challenge_expires_at"]) <= mfa.now():
        raise HTTPException(status_code=401, detail="本次登录已过期，请重新登录。")

    await mfa.verify_code(row["user_id"], row, req.code)
    await db.clear_2fa_challenge(row["user_id"])
    user = await get_user_by_id(row["user_id"])
    token = await create_session(user["id"])
    return {"token": token, "user": _public_user(user)}


@app.post("/api/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        await delete_session(token)
    return {"status": "ok"}


@app.get("/api/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {"user": _public_user(user)}


class UpdateEmailRequest(BaseModel):
    email: str


@app.put("/api/auth/email")
async def update_email(req: UpdateEmailRequest, user: dict = Depends(get_current_user)):
    email = _normalize_email(req.email)
    existing = await get_user_by_email(email)
    if existing and existing["id"] != user["id"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="邮箱已注册")
    await update_user_email(user["id"], email)
    return {"ok": True}


class UpdatePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@app.put("/api/auth/password")
async def update_password(req: UpdatePasswordRequest, user: dict = Depends(get_current_user)):
    row = await db.ensure_2fa_row(user["id"])
    await _reauthenticate(user, row, req.current_password)
    if len(req.new_password) < 4:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="密码至少需要 4 个字符")
    await update_user_password(user["id"], req.new_password)
    token = await _rotate_session(user)
    return {"token": token, "user": _public_user(user)}


class DeleteAccountRequest(BaseModel):
    password: str = Field(min_length=1)
    confirmation: str
    code: Optional[str] = None


@app.delete("/api/auth/account")
async def delete_account(req: DeleteAccountRequest, user: dict = Depends(get_current_user)):
    if req.confirmation.strip() != user["username"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请准确输入用户名以确认删除账户",
        )

    row = await db.ensure_2fa_row(user["id"])
    await _reauthenticate(user, row, req.password)
    if row["enabled"]:
        if not req.code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="请输入双重认证验证码或恢复码",
            )
        await mfa.verify_code(user["id"], row, req.code)

    services = await get_user_services(user["id"])
    service_ids = [service["id"] for service in services]
    if not await delete_user_account(user["id"]):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="账户不存在")
    _clear_service_runtime_state(service_ids)
    return {"status": "ok"}


# ── OIDC / SSO ──

@app.get("/api/oidc/config")
async def oidc_config():
    return {
        "enabled": oidc.OIDC_ENABLED,
        "provider_name": oidc.OIDC_PROVIDER_NAME,
        "disable_password_login": oidc.OIDC_DISABLE_PASSWORD_LOGIN,
    }


@app.get("/api/oidc/authorize")
async def oidc_authorize():
    if not oidc.OIDC_ENABLED:
        raise HTTPException(status_code=400, detail="未启用 OIDC")
    state = await create_oidc_state()
    return RedirectResponse(oidc.build_auth_url(state))


@app.get("/api/oidc/callback")
async def oidc_callback(code: Optional[str] = None, state: Optional[str] = None):
    if not oidc.OIDC_ENABLED:
        raise HTTPException(status_code=400, detail="未启用 OIDC")
    try:
        if not code or not state or not await consume_oidc_state(state):
            raise HTTPException(status_code=400, detail="登录请求无效或已过期")

        userinfo = oidc.fetch_userinfo(code)
        identifier = userinfo.get(oidc.OIDC_IDENTIFIER_FIELD)
        if not identifier:
            raise HTTPException(
                status_code=400, detail=f"用户信息中未找到字段 '{oidc.OIDC_IDENTIFIER_FIELD}'"
            )

        user = await get_user_by_email(str(identifier))
        if not user:
            if not oidc.OIDC_AUTO_CREATE_USERS:
                raise HTTPException(status_code=403, detail="此身份没有关联账户")

            raw_email = userinfo.get("email", str(identifier))
            raw_username = (
                userinfo.get("preferred_username")
                or userinfo.get("name")
                or (raw_email.split("@")[0] if "@" in raw_email else raw_email)
            )
            username = str(raw_username).strip().lower()[:64] or "user"

            base, counter = username, 1
            while await get_user_by_username(username):
                username = f"{base}{counter}"
                counter += 1

            user_id = await create_oidc_user(username, str(identifier))
            user = await get_user_by_id(user_id)

        challenge_token = await mfa.begin_login(user["id"])
        if challenge_token:
            return RedirectResponse(f"/oidc-callback?challenge_token={challenge_token}")

        token = await create_session(user["id"])
        return RedirectResponse(f"/oidc-callback?token={token}")
    except HTTPException as e:
        return RedirectResponse(f"/login?oidc_error={quote(e.detail)}")


# ── Password reset ──

class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


@app.get("/api/auth/password-reset-status")
async def password_reset_status():
    return {"enabled": bool(email_notifier.SMTP_ADDRESS)}


@app.post("/api/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    """Always returns 200, regardless of whether the email is registered, so
    this endpoint can't be used to find out who has an account."""
    if not email_notifier.SMTP_ADDRESS:
        raise HTTPException(status_code=503, detail="未配置密码重置功能")

    email = _normalize_email(req.email)
    user = await get_user_by_email(email)
    if user:
        token = await create_password_reset_token(user["id"])
        link = f"{SERVER_URL}/reset-password?token={token}"
        body = (
            "我们收到了重置 OpenUsage 密码的请求。\n\n"
            f"请通过此链接重置密码：{link}\n\n"
            "链接将在 1 小时后过期。如果不是你本人操作，请忽略此邮件。"
        )
        try:
            email_notifier.send(user["email"], "重置 OpenUsage 密码", body)
        except email_notifier.EmailError as e:
            logger.error(f"Failed to send password reset email to {user['email']}: {e}")

    return {"message": "如果该邮箱已注册，我们已发送密码重置链接。"}


@app.post("/api/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    record = await get_password_reset_token(req.token)
    if not record:
        raise HTTPException(status_code=400, detail="重置链接无效或已使用")

    created_at = datetime.fromisoformat(record["created_at"])
    if datetime.now(timezone.utc) - created_at > PASSWORD_RESET_MAX_AGE:
        await delete_password_reset_token(req.token)
        raise HTTPException(status_code=400, detail="重置链接已过期")

    if len(req.new_password) < 4:
        raise HTTPException(status_code=400, detail="密码至少需要 4 个字符")

    await update_user_password(record["user_id"], req.new_password)
    await delete_all_sessions_for_user(record["user_id"])
    await delete_password_reset_token(req.token)
    return {"ok": True}


# ── Two-factor authentication ──

class TwoFactorPasswordRequest(BaseModel):
    password: str


class TwoFactorCodeRequest(BaseModel):
    code: str


class TwoFactorManageRequest(BaseModel):
    password: str
    code: str


async def _reauthenticate(user: dict, row: dict, password: str):
    mfa.check_limit(row)
    if not verify_password(password, user["password_hash"]):
        await mfa.fail(user["id"], row, "Incorrect password")


async def _rotate_session(user: dict) -> str:
    await delete_all_sessions_for_user(user["id"])
    return await create_session(user["id"])


@app.get("/api/auth/2fa")
async def two_factor_status(user: dict = Depends(get_current_user)):
    row = await db.get_2fa(user["id"])
    return {
        "enabled": bool(row and row["enabled"]),
        "recovery_codes_remaining": len(row["recovery_hashes"]) if row else 0,
    }


@app.post("/api/auth/2fa/setup")
async def two_factor_setup(req: TwoFactorPasswordRequest, user: dict = Depends(get_current_user)):
    row = await db.ensure_2fa_row(user["id"])
    await _reauthenticate(user, row, req.password)
    if row["enabled"]:
        raise HTTPException(status_code=400, detail="双重认证已启用")

    secret = pyotp.random_base32()
    pending_secret = mfa.cipher().encrypt(secret.encode()).decode()
    await db.set_2fa_pending(user["id"], pending_secret, (mfa.now() + timedelta(minutes=10)).isoformat())

    uri = pyotp.TOTP(secret).provisioning_uri(name=user["username"], issuer_name="OpenUsage")
    image = io.BytesIO()
    qrcode.make(uri).save(image, format="PNG")
    return {"secret": secret, "qr_code": "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()}


@app.post("/api/auth/2fa/enable")
async def two_factor_enable(req: TwoFactorCodeRequest, user: dict = Depends(get_current_user)):
    row = await db.ensure_2fa_row(user["id"])
    mfa.check_limit(row)
    if row["enabled"] or not row["pending_secret"] or not row["pending_expires_at"] or \
            datetime.fromisoformat(row["pending_expires_at"]) <= mfa.now():
        raise HTTPException(status_code=400, detail="请重新开始设置双重认证")

    pending_secret = mfa.cipher().decrypt(row["pending_secret"].encode()).decode()
    step = mfa.totp_step(pending_secret, req.code.strip())
    if step is None:
        await mfa.fail(user["id"], row)

    codes, hashes = mfa.recovery_codes()
    await db.enable_2fa(user["id"], row["pending_secret"], hashes, step)
    token = await _rotate_session(user)
    return {"recovery_codes": codes, "token": token, "user": _public_user(user)}


@app.post("/api/auth/2fa/disable")
async def two_factor_disable(req: TwoFactorManageRequest, user: dict = Depends(get_current_user)):
    row = await db.ensure_2fa_row(user["id"])
    await _reauthenticate(user, row, req.password)
    if not row["enabled"]:
        raise HTTPException(status_code=400, detail="未启用双重认证")
    await mfa.verify_code(user["id"], row, req.code)
    await db.disable_2fa(user["id"])
    token = await _rotate_session(user)
    return {"token": token, "user": _public_user(user)}


@app.post("/api/auth/2fa/recovery-codes")
async def two_factor_recovery_codes(req: TwoFactorManageRequest, user: dict = Depends(get_current_user)):
    row = await db.ensure_2fa_row(user["id"])
    await _reauthenticate(user, row, req.password)
    if not row["enabled"]:
        raise HTTPException(status_code=400, detail="未启用双重认证")
    await mfa.verify_code(user["id"], row, req.code)
    codes, hashes = mfa.recovery_codes()
    await db.update_2fa_recovery_hashes(user["id"], hashes)
    return {"recovery_codes": codes}


# User Services CRUD Endpoints
@app.get("/api/user-services")
async def list_user_services(user: dict = Depends(get_current_user)):
    services = await get_user_services(user["id"])
    return services


class UserServiceCreate(BaseModel):
    service_type: str
    name: str
    config: dict
    category_id: Optional[int] = None
    notify_thresholds: Optional[dict] = None
    refresh_interval_minutes: int = Field(
        default=DEFAULT_REFRESH_INTERVAL_MINUTES, ge=1, le=10080
    )
    store_history: bool = False


@app.post("/api/user-services")
async def create_user_service(req: UserServiceCreate, user: dict = Depends(get_current_user)):
    if req.service_type not in SERVICE_FETCHERS:
        raise HTTPException(status_code=400, detail=f"无效的服务类型：{req.service_type}")
    svc_name = req.name.strip() or req.service_type.capitalize()
    svc_id = await add_user_service(
        user["id"],
        req.service_type,
        svc_name,
        req.config,
        req.category_id,
        req.notify_thresholds,
        req.refresh_interval_minutes,
        req.store_history,
    )
    return {"status": "ok", "id": svc_id}


class UserServiceUpdate(BaseModel):
    name: str
    config: dict
    category_id: Optional[int] = None
    # None means "not part of this update" so category moves, cURL refreshes,
    # and provider credential rotation preserve the existing thresholds.
    notify_thresholds: Optional[dict] = None
    # None likewise preserves the cadence for older clients and internal
    # config/category-only updates.
    refresh_interval_minutes: Optional[int] = Field(default=None, ge=1, le=10080)
    # None preserves the stored flag, same as the fields above.
    store_history: Optional[bool] = None


@app.put("/api/user-services/{service_id}")
async def update_service(
    service_id: int, req: UserServiceUpdate, user: dict = Depends(get_current_user)
):
    existing = await get_user_service(service_id, user["id"])
    if not existing:
        raise HTTPException(status_code=404, detail="服务不存在")
    svc_name = req.name.strip() or existing["service_type"].capitalize()
    effective_thresholds = (
        existing.get("notify_thresholds", {})
        if req.notify_thresholds is None
        else req.notify_thresholds
    )
    effective_refresh_interval = (
        existing.get("refresh_interval_minutes", DEFAULT_REFRESH_INTERVAL_MINUTES)
        if req.refresh_interval_minutes is None
        else req.refresh_interval_minutes
    )
    await update_user_service(
        service_id,
        user["id"],
        svc_name,
        req.config,
        req.category_id,
        req.notify_thresholds,
        req.refresh_interval_minutes,
        req.store_history,
    )
    _last_fetch_error.pop(service_id, None)
    updated_svc = {
        **existing,
        "name": svc_name,
        "config": req.config,
        "category_id": req.category_id,
        "notify_thresholds": effective_thresholds,
        "refresh_interval_minutes": effective_refresh_interval,
        "store_history": (
            existing.get("store_history", False)
            if req.store_history is None
            else req.store_history
        ),
    }
    _trigger_background_live_fetch(updated_svc)
    return {"status": "ok"}


@app.get("/api/user-services/{service_id}/history")
async def get_service_history(
    service_id: int,
    since: Optional[str] = None,
    limit: int = Query(default=10000, ge=1, le=10000),
    max_points: int = Query(default=400, ge=10, le=2000),
    user: dict = Depends(get_current_user),
):
    # Summaries are computed over every fetched entry in the selected range
    # (no silent truncation); the chart itself receives downsampled points
    # (stride sampling that always keeps the newest point), plus coverage
    # metadata so a truncated response is visible instead of misleading.
    existing = await get_user_service(service_id, user["id"])
    if not existing:
        raise HTTPException(status_code=404, detail="服务不存在")
    history = await get_service_usage_history(service_id, since, limit)
    total_entries = await count_service_usage_history(service_id, since)
    truncated = total_entries > len(history)
    # Pre-extract the chartable figures so the client doesn't have to know
    # every provider's payload shape.
    series: dict = {}
    for entry in history:
        metrics = extract_history_metrics(existing["service_type"], entry["data"])
        for key, m in metrics.items():
            s = series.setdefault(key, {"points": [], "values": []})
            s.update(label=m["label"], unit=m["unit"])
            s["points"].append({"t": entry["fetched_at"], "v": m["value"]})
            s["values"].append(m["value"])
    for s in series.values():
        values = s.pop("values")
        count = len(values)
        s["summary"] = {
            "count": count,
            "avg": round(sum(values) / count, 4) if count else None,
            "min": min(values) if count else None,
            "max": max(values) if count else None,
        }
        if count > max_points:
            stride = -(-(count - 1) // (max_points - 1))  # ceil: keeps ≤ max_points points
            indices = list(range(0, count - 1, stride))
            indices.append(count - 1)  # always keep the newest point
            s["points"] = [s["points"][i] for i in indices]
    first = history[0]["fetched_at"] if history else None
    last = history[-1]["fetched_at"] if history else None
    return {
        "store_history": existing["store_history"],
        "service_type": existing["service_type"],
        "series": series,
        "coverage": {
            "first": first,
            "last": last,
            "total_entries": total_entries,
            "truncated": truncated,
        },
    }


@app.delete("/api/user-services/{service_id}")
async def delete_service(service_id: int, user: dict = Depends(get_current_user)):
    existing = await get_user_service(service_id, user["id"])
    if not existing:
        raise HTTPException(status_code=404, detail="服务不存在")
    await delete_user_service(service_id, user["id"])
    _clear_service_runtime_state([service_id])
    return {"status": "ok"}


class DeleteAllServicesRequest(BaseModel):
    confirmation: str


@app.delete("/api/user-services")
async def delete_all_services(
    req: DeleteAllServicesRequest, user: dict = Depends(get_current_user)
):
    if req.confirmation.strip() != "DELETE ALL":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='请输入“DELETE ALL”以确认',
        )
    services = await get_user_services(user["id"])
    service_ids = [service["id"] for service in services]
    deleted_count = await delete_all_user_services(user["id"])
    _clear_service_runtime_state(service_ids)
    return {"status": "ok", "deleted_count": deleted_count}


# Category CRUD Endpoints (home-page sections that services can be assigned to)
@app.get("/api/categories")
async def list_categories(user: dict = Depends(get_current_user)):
    return await get_categories(user["id"])


class CategoryCreate(BaseModel):
    name: str


@app.post("/api/categories")
async def create_category_endpoint(req: CategoryCreate, user: dict = Depends(get_current_user)):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="名称不能为空")
    cat_id = await create_category(user["id"], name)
    return {"status": "ok", "id": cat_id}


class CategoryUpdate(BaseModel):
    name: str


@app.put("/api/categories/{category_id}")
async def update_category_endpoint(
    category_id: int, req: CategoryUpdate, user: dict = Depends(get_current_user)
):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="名称不能为空")
    updated = await update_category(category_id, user["id"], name)
    if not updated:
        raise HTTPException(status_code=404, detail="分类不存在")
    return {"status": "ok"}


@app.delete("/api/categories/{category_id}")
async def delete_category_endpoint(category_id: int, user: dict = Depends(get_current_user)):
    await delete_category(category_id, user["id"])
    return {"status": "ok"}


class CategoryMove(BaseModel):
    direction: str  # "up" or "down"


@app.post("/api/categories/{category_id}/move")
async def move_category_endpoint(
    category_id: int, req: CategoryMove, user: dict = Depends(get_current_user)
):
    if req.direction not in ("up", "down"):
        raise HTTPException(status_code=400, detail="移动方向必须是 'up' 或 'down'")
    moved = await move_category(category_id, user["id"], req.direction)
    if not moved:
        raise HTTPException(status_code=400, detail="无法继续向该方向移动分类")
    return {"status": "ok"}


# Usage Endpoints
@app.get("/api/usage")
async def get_all_usage(force: bool = False, user: dict = Depends(get_current_user)):
    user_services = await get_user_services(user["id"])
    if not user_services:
        return []
    results = await asyncio.gather(
        *(_fetch_and_save_user_service(s, force_refresh=force) for s in user_services)
    )
    return list(results)


@app.get("/api/usage/{service_id}")
async def get_one_usage(
    service_id: int, force: bool = False, user: dict = Depends(get_current_user)
):
    svc = await get_user_service(service_id, user["id"])
    if not svc:
        raise HTTPException(status_code=404, detail="服务不存在")
    return await _fetch_and_save_user_service(svc, force_refresh=force)


# Import cURL Helper Endpoints
class CurlItem(BaseModel):
    curl: str


@app.post("/api/import-curl/claude")
async def claude_import_curl(item: CurlItem):
    try:
        parsed = parse_curl(item.curl)
        return {"status": "ok", "org_url": parsed["url"], "cookies": parsed["cookies"]}
    except ValueError as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/import-curl/gemini")
async def gemini_import_curl(item: CurlItem):
    try:
        parsed = parse_gemini_curl(item.curl)
        return {
            "status": "ok",
            "cookies": parsed["cookies"],
            "at_token": parsed["at_token"],
            "bl": parsed["bl"],
            "sid": parsed["sid"],
        }
    except ValueError as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/import-curl/openai")
async def openai_import_curl(item: CurlItem):
    try:
        parsed = parse_openai_session_curl(item.curl)
        return {"status": "ok", "session_key": parsed["session_key"], "org_id": parsed["org_id"]}
    except ValueError as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/import-curl/chatgpt")
async def chatgpt_import_curl(item: CurlItem):
    try:
        parsed = parse_chatgpt_curl(item.curl)
        return {"status": "ok", "cookies": parsed["cookies"]}
    except ValueError as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/import-curl/ollama_cloud")
async def ollama_cloud_import_curl(item: CurlItem):
    try:
        parsed = parse_ollama_curl(item.curl)
        return {"status": "ok", "cookies": parsed["cookies"]}
    except ValueError as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/import-curl/opencode")
async def opencode_import_curl(item: CurlItem):
    try:
        parsed = parse_opencode_curl(item.curl)
        return {"status": "ok", "cookies": parsed["cookies"], "org_id": parsed["org_id"]}
    except ValueError as e:
        return {"status": "error", "error": str(e)}


class GenericTestRequest(BaseModel):
    config: dict


@app.post("/api/test-custom")
async def test_custom_endpoint(req: GenericTestRequest, user: dict = Depends(get_current_user)):
    return await test_generic(req.config)


# Copilot Device Flow Endpoints. Authorization sessions are stored per user
# (keyed by user id + server-generated flow id) with an expiry taken from
# GitHub's expires_in; terminal poll results clean up immediately. The
# in-memory store means a backend restart cancels any pending flow — the
# frontend just starts a new one, which matches the 15-minute lifetime of a
# device code anyway.
_copilot_flows: dict[tuple[int, str], dict] = {}


def _copilot_flow_get(user_id: int, flow_id: str) -> Optional[dict]:
    flow = _copilot_flows.get((user_id, flow_id))
    if flow is None:
        return None
    if time.time() > flow["expires_at"]:
        _copilot_flows.pop((user_id, flow_id), None)
        return None
    return flow


@app.post("/api/copilot/device/start")
async def copilot_device_start(user: dict = Depends(get_current_user)):
    data = await start_device_flow()
    flow_id = uuid.uuid4().hex[:12]
    _copilot_flows[(user["id"], flow_id)] = {
        "device_code": data["device_code"],
        "interval": data["interval"],
        "expires_at": time.time() + data["expires_in"],
    }
    return {**data, "flow_id": flow_id}


class CopilotPollItem(BaseModel):
    flow_id: str


@app.post("/api/copilot/device/poll")
async def copilot_device_poll(req: CopilotPollItem, user: dict = Depends(get_current_user)):
    flow = _copilot_flow_get(user["id"], req.flow_id)
    if not flow:
        return {"status": "expired"}

    result = await poll_device_flow(flow["device_code"])
    if result["status"] == "pending":
        if "interval" in result:
            flow["interval"] = result["interval"]
        return result

    # Terminal states (ok / expired / denied / error) end the session.
    _copilot_flows.pop((user["id"], req.flow_id), None)
    return result


# Push notification endpoints. There's no separate "enabled" flag on the
# user -- subscribing a device on the Settings page is what turns push
# notifications on for that user; unsubscribing every device turns it off.
@app.get("/api/push/public-key")
async def push_public_key():
    return {"public_key": push_notifier.VAPID_PUBLIC_KEY if push_notifier.is_configured() else None}


class PushSubscriptionKeys(BaseModel):
    p256dh: str
    auth: str


class PushSubscriptionCreate(BaseModel):
    endpoint: str
    keys: PushSubscriptionKeys


class PushSubscriptionDelete(BaseModel):
    endpoint: str


@app.post("/api/push/subscribe")
async def push_subscribe(req: PushSubscriptionCreate, user: dict = Depends(get_current_user)):
    if not push_notifier.is_configured():
        raise HTTPException(status_code=400, detail="此实例未配置推送通知")
    await add_push_subscription(user["id"], req.endpoint, req.keys.p256dh, req.keys.auth)
    return {"status": "ok"}


@app.post("/api/push/unsubscribe")
async def push_unsubscribe(req: PushSubscriptionDelete, user: dict = Depends(get_current_user)):
    await delete_push_subscription(user["id"], req.endpoint)
    return {"status": "ok"}


@app.post("/api/push/test")
async def push_test(user: dict = Depends(get_current_user)):
    try:
        await push_notifier.send_push(
            user["id"], "OpenUsage 测试通知", "如果看到了这条消息，说明推送通知工作正常。"
        )
    except push_notifier.PushNotifierError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}


# Serve the built frontend. Mounted last so it never shadows the /api/ routes
# above; html=True gives directory index.html lookup and a 404.html fallback,
# matching the previous nginx try_files/error_page behavior.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
