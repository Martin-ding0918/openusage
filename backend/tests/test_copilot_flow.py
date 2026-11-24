"""Copilot 设备授权流的用户隔离与有效期测试。

覆盖：匿名请求 401；flow 状态按 (user, flow) 隔离；终态（成功/拒绝/过期/错误）
立即清理；expires_at 过期视为 expired 并清理；两个账号互不干扰。
"""

import importlib
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import db
from fastapi.testclient import TestClient


def _import_backend_main():
    """main.py mounts ./static at import time, so import it from backend/."""
    backend_dir = Path(__file__).resolve().parents[1]
    previous_cwd = os.getcwd()
    try:
        os.chdir(backend_dir)
        return importlib.import_module("main")
    finally:
        os.chdir(previous_cwd)


backend_main = _import_backend_main()

START_DATA = {
    "device_code": "DEV-123",
    "user_code": "ABCD-1234",
    "verification_uri": "https://github.com/login/device",
    "expires_in": 900,
    "interval": 5,
}


class CopilotFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="openusage-copilot-test-")
        self.original_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self.temp_dir.name, "test.db")
        await db.init_db()
        backend_main._copilot_flows.clear()
        self.user_id = await db.create_user("copilot-test-user", "test-password")
        self.other_user_id = await db.create_user("copilot-other-user", "test-password")
        self.token = await db.create_session(self.user_id)
        self.other_token = await db.create_session(self.other_user_id)
        self.client = TestClient(backend_main.app)

    async def asyncTearDown(self):
        db.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def _auth(self, token):
        return {"Authorization": f"Bearer {token}"}

    async def test_start_and_poll_require_auth(self):
        r1 = self.client.post("/api/copilot/device/start")
        self.assertEqual(r1.status_code, 401)
        r2 = self.client.post("/api/copilot/device/poll", json={"flow_id": "x"})
        self.assertEqual(r2.status_code, 401)
        self.assertEqual(len(backend_main._copilot_flows), 0)

    async def test_start_returns_flow_id_and_stores_per_user(self):
        with patch.object(backend_main, "start_device_flow", new=AsyncMock(return_value=START_DATA)):
            r = self.client.post("/api/copilot/device/start", headers=self._auth(self.token))
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("flow_id", body)
        self.assertEqual(body["device_code"], "DEV-123")
        flow_key = (self.user_id, body["flow_id"])
        self.assertIn(flow_key, backend_main._copilot_flows)
        self.assertEqual(backend_main._copilot_flows[flow_key]["device_code"], "DEV-123")

    async def test_poll_success_cleans_up_immediately(self):
        with patch.object(backend_main, "start_device_flow", new=AsyncMock(return_value=START_DATA)):
            flow_id = self.client.post(
                "/api/copilot/device/start", headers=self._auth(self.token)
            ).json()["flow_id"]
        with patch.object(
            backend_main,
            "poll_device_flow",
            new=AsyncMock(return_value={"status": "ok", "token": "gho_secret"}),
        ):
            r = self.client.post(
                "/api/copilot/device/poll",
                headers=self._auth(self.token),
                json={"flow_id": flow_id},
            )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["token"], "gho_secret")
        self.assertNotIn((self.user_id, flow_id), backend_main._copilot_flows)

    async def test_poll_isolated_between_users(self):
        with patch.object(backend_main, "start_device_flow", new=AsyncMock(return_value=START_DATA)):
            flow_id = self.client.post(
                "/api/copilot/device/start", headers=self._auth(self.token)
            ).json()["flow_id"]
        # 另一个账号拿同一个 flow_id 轮询：自己的命名空间里没有 → expired，
        # 且绝不能拿到首个用户的 device_code / token。
        with patch.object(
            backend_main,
            "poll_device_flow",
            new=AsyncMock(return_value={"status": "ok", "token": "gho_stolen"}),
        ):
            r = self.client.post(
                "/api/copilot/device/poll",
                headers=self._auth(self.other_token),
                json={"flow_id": flow_id},
            )
        self.assertEqual(r.json()["status"], "expired")
        self.assertNotIn("token", r.json())

    async def test_expired_flow_is_rejected_and_cleaned(self):
        with patch.object(backend_main, "start_device_flow", new=AsyncMock(return_value=START_DATA)):
            flow_id = self.client.post(
                "/api/copilot/device/start", headers=self._auth(self.token)
            ).json()["flow_id"]
        backend_main._copilot_flows[(self.user_id, flow_id)]["expires_at"] = time.time() - 1
        with patch.object(backend_main, "poll_device_flow", new=AsyncMock(return_value={"status": "ok", "token": "t"})):
            r = self.client.post(
                "/api/copilot/device/poll",
                headers=self._auth(self.token),
                json={"flow_id": flow_id},
            )
        self.assertEqual(r.json()["status"], "expired")
        self.assertNotIn("token", r.json())
        self.assertNotIn((self.user_id, flow_id), backend_main._copilot_flows)

    async def test_denied_flow_cleans_up(self):
        with patch.object(backend_main, "start_device_flow", new=AsyncMock(return_value=START_DATA)):
            flow_id = self.client.post(
                "/api/copilot/device/start", headers=self._auth(self.token)
            ).json()["flow_id"]
        with patch.object(backend_main, "poll_device_flow", new=AsyncMock(return_value={"status": "denied"})):
            r = self.client.post(
                "/api/copilot/device/poll",
                headers=self._auth(self.token),
                json={"flow_id": flow_id},
            )
        self.assertEqual(r.json()["status"], "denied")
        self.assertNotIn((self.user_id, flow_id), backend_main._copilot_flows)

    async def test_pending_flow_keeps_state_and_updates_interval(self):
        with patch.object(backend_main, "start_device_flow", new=AsyncMock(return_value=START_DATA)):
            flow_id = self.client.post(
                "/api/copilot/device/start", headers=self._auth(self.token)
            ).json()["flow_id"]
        with patch.object(
            backend_main,
            "poll_device_flow",
            new=AsyncMock(return_value={"status": "pending", "interval": 10}),
        ):
            r = self.client.post(
                "/api/copilot/device/poll",
                headers=self._auth(self.token),
                json={"flow_id": flow_id},
            )
        self.assertEqual(r.json(), {"status": "pending", "interval": 10})
        self.assertEqual(
            backend_main._copilot_flows[(self.user_id, flow_id)]["interval"], 10
        )


class _FakeResp:
    """模拟 GitHub OAuth 响应（httpx.Response 的最小面）。"""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """httpx.AsyncClient 替身：类级脚本+游标，跨实例按顺序消费模拟 GitHub 响应。"""

    script = []
    cursor = 0

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kw):
        resp = _FakeAsyncClient.script[_FakeAsyncClient.cursor]
        _FakeAsyncClient.cursor += 1
        return _FakeResp(resp)


class CopilotFullFlowGitHubMockTests(unittest.IsolatedAsyncioTestCase):
    """模拟 GitHub 响应走真 copilot.py 的完整启动→轮询流程（不 patch 业务函数）。"""

    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="openusage-copilot-gh-test-")
        self.original_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self.temp_dir.name, "test.db")
        await db.init_db()
        backend_main._copilot_flows.clear()
        self.user_id = await db.create_user("copilot-gh-user", "test-password")
        self.other_user_id = await db.create_user("copilot-gh-other", "test-password")
        self.token = await db.create_session(self.user_id)
        self.other_token = await db.create_session(self.other_user_id)
        self.client = TestClient(backend_main.app)
        _FakeAsyncClient.script = [
            START_DATA,                                    # start：GitHub 下发 device_code
            {"error": "authorization_pending"},            # poll 1：等待授权
            {"error": "slow_down", "interval": 10},        # poll 2：要求放慢
            {"access_token": "gho_real", "token_type": "bearer"},  # poll 3：授权成功
        ]
        _FakeAsyncClient.cursor = 0

    async def asyncTearDown(self):
        db.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def _auth(self, token):
        return {"Authorization": f"Bearer {token}"}

    async def test_full_flow_with_mocked_github_and_isolation(self):
        headers = self._auth(self.token)
        with patch("fetchers.copilot.httpx.AsyncClient", _FakeAsyncClient):
            r1 = self.client.post("/api/copilot/device/start", headers=headers)
        self.assertEqual(r1.status_code, 200)
        body = r1.json()
        self.assertEqual(body["device_code"], "DEV-123")
        flow_id = body["flow_id"]
        flow_key = (self.user_id, flow_id)
        self.assertIn(flow_key, backend_main._copilot_flows)

        headers_other = self._auth(self.other_token)
        with patch("fetchers.copilot.httpx.AsyncClient", _FakeAsyncClient):
            p1 = self.client.post("/api/copilot/device/poll", headers=headers,
                                  json={"flow_id": flow_id})
            # 另一账号拿 A 的 flow_id：B 的命名空间无此会话 → expired，碰不到 GitHub
            p_other = self.client.post("/api/copilot/device/poll",
                                       headers=headers_other, json={"flow_id": flow_id})
            p2 = self.client.post("/api/copilot/device/poll", headers=headers,
                                  json={"flow_id": flow_id})
            p3 = self.client.post("/api/copilot/device/poll", headers=headers,
                                  json={"flow_id": flow_id})

        self.assertEqual(p1.json(), {"status": "pending"})
        self.assertEqual(p_other.json()["status"], "expired")  # 隔离：B 无 A 的会话
        self.assertNotIn("token", p_other.json())
        self.assertEqual(p2.json(), {"status": "pending", "interval": 10})
        final = p3.json()
        self.assertEqual(final["status"], "ok")
        self.assertEqual(final["token"], "gho_real")  # token 只返回给会话所有者
        self.assertNotIn(flow_key, backend_main._copilot_flows)  # 成功后立即清理


if __name__ == "__main__":
    unittest.main()
