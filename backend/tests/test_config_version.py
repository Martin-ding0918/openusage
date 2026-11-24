"""配置版本化测试：update_user_service 递增版本；刷新期间配置变更时旧任务
丢弃结果与回写、按新配置重刷（P2「后台刷新覆盖新配置」）。"""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import db


def _import_backend_main():
    import importlib

    backend_dir = Path(__file__).resolve().parents[1]
    previous_cwd = os.getcwd()
    try:
        os.chdir(backend_dir)
        return importlib.import_module("main")
    finally:
        os.chdir(previous_cwd)


backend_main = _import_backend_main()


class ConfigVersionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="openusage-configver-test-")
        self.original_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self.temp_dir.name, "test.db")
        await db.init_db()
        self.user_id = await db.create_user("configver-user", "test-password")
        self.service_id = await db.add_user_service(
            self.user_id, "custom", "Private API", {"api_key": "secret"}
        )

    async def asyncTearDown(self):
        db.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    async def test_version_starts_at_one(self):
        svc = await db.get_user_service(self.service_id, self.user_id)
        self.assertEqual(svc["config_version"], 1)

    async def test_update_without_thresholds_increments(self):
        await db.update_user_service(self.service_id, self.user_id, "Renamed", {"api_key": "new"})
        svc = await db.get_user_service(self.service_id, self.user_id)
        self.assertEqual(svc["config_version"], 2)
        self.assertEqual(svc["name"], "Renamed")

    async def test_update_with_thresholds_increments(self):
        await db.update_user_service(
            self.service_id, self.user_id, "Private API", {"api_key": "same"},
            notify_thresholds={"monthly_threshold": 50},
        )
        svc = await db.get_user_service(self.service_id, self.user_id)
        self.assertEqual(svc["config_version"], 2)
        self.assertEqual(svc["notify_thresholds"], {"monthly_threshold": 50})

    async def test_consecutive_updates_accumulate(self):
        await db.update_user_service(self.service_id, self.user_id, "A1", {"v": 1})
        await db.update_user_service(self.service_id, self.user_id, "A2", {"v": 2})
        svc = await db.get_user_service(self.service_id, self.user_id)
        self.assertEqual(svc["config_version"], 3)


class StaleFetchDiscardTests(unittest.IsolatedAsyncioTestCase):
    """刷新任务运行期间用户改配置：结果与回写被丢弃，按新配置重刷。"""

    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="openusage-stalefetch-test-")
        self.original_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self.temp_dir.name, "test.db")
        await db.init_db()
        self.user_id = await db.create_user("stale-user", "test-password")
        self.service_id = await db.add_user_service(
            self.user_id, "custom", "Private API", {"api_key": "secret"}
        )
        # 任务捕获的快照（version 1）
        self.snapshot = dict(await db.get_user_service(self.service_id, self.user_id))
        self.fake_result = {
            "configured": True,
            "status": "ok",
            "data": {"fields": [{
                "path": "data.quota", "label": "余额", "symbol": "$",
                "type": "balance", "value": 42, "history": True,
            }]},
        }

    async def asyncTearDown(self):
        db.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    async def _history_count(self) -> int:
        return len(await db.get_service_usage_history(self.service_id, None, None))

    async def test_unchanged_config_saves_normally(self):
        fake_fetcher = AsyncMock(return_value=dict(self.fake_result))
        with patch.dict(backend_main.SERVICE_FETCHERS, {"custom": fake_fetcher}):
            result = await backend_main._live_fetch_and_save(dict(self.snapshot))

        self.assertNotIn("discarded_config_changed", result["result"])
        self.assertEqual(await self._history_count(), 1)  # 正常写入历史
        svc_now = await db.get_user_service(self.service_id, self.user_id)
        self.assertEqual(svc_now["name"], "Private API")
        self.assertEqual(svc_now["config_version"], 1)
