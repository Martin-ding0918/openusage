import httpx

# Device flow client id used by VS Code's Copilot Chat extension. GitHub does not
# document a public API for personal Copilot subscription status, so this replicates
# the same OAuth device authorization flow official editor extensions use to sign in.
CLIENT_ID = "Iv1.b507a08c87ecfe98"
DEVICE_CODE_URL = "https://github.com/login/device/code"
ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
USAGE_URL = "https://api.github.com/copilot_internal/user"

TOKEN_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "GitHubCopilotChat/0.26.7",
    "Editor-Version": "vscode/1.104.1",
}


async def start_device_flow() -> dict:
    """Start a GitHub device authorization flow. Pure — no server-side state;
    the caller owns per-user flow state (see main._copilot_flows)."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            DEVICE_CODE_URL,
            data={"client_id": CLIENT_ID, "scope": "read:user"},
            headers={"Accept": "application/json"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()

    return {
        "device_code": data["device_code"],
        "user_code": data["user_code"],
        "verification_uri": data["verification_uri"],
        "expires_in": data.get("expires_in", 900),
        "interval": data.get("interval", 5),
    }


async def poll_device_flow(device_code: str) -> dict:
    """Pure GitHub token polling; the caller owns per-user flow state and is
    responsible for cleaning it up on terminal results."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            ACCESS_TOKEN_URL,
            data={
                "client_id": CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()

    error = data.get("error")
    if error == "authorization_pending":
        return {"status": "pending"}
    if error == "slow_down":
        return {"status": "pending", "interval": data.get("interval", 5)}
    if error == "expired_token":
        return {"status": "expired"}
    if error == "access_denied":
        return {"status": "denied"}
    if error:
        return {"status": "error", "error": error}
    return {"status": "ok", "token": data.get("access_token")}


def _quota(q: dict) -> dict:
    return {
        "unlimited": q.get("unlimited", False),
        "remaining": q.get("remaining"),
        "entitlement": q.get("entitlement"),
        "percent_remaining": q.get("percent_remaining"),
    }


async def fetch_copilot(config: dict = None) -> dict:
    if config is None:
        config = {}
    token = config.get("copilot_token")

    if not token:
        return {"configured": False, "error": "Not configured"}

    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                USAGE_URL,
                headers={**TOKEN_HEADERS, "Authorization": f"token {token}"},
                timeout=10,
            )
            r.raise_for_status()
            info = r.json()
            quotas = info.get("quota_snapshots", {})
            return {
                "configured": True,
                "status": "ok",
                "data": {
                    "plan": info.get("copilot_plan"),
                    "quota_reset_date": info.get("quota_reset_date"),
                    "premium_requests": _quota(quotas.get("premium_interactions", {})),
                    "chat": _quota(quotas.get("chat", {})),
                    "completions": _quota(quotas.get("completions", {})),
                },
            }
        except httpx.HTTPStatusError as e:
            return {
                "configured": True,
                "status": "error",
                "error": f"HTTP {e.response.status_code} — sign-in may have expired, reconnect in Settings",
            }
        except Exception as e:
            return {"configured": True, "status": "error", "error": str(e)}
