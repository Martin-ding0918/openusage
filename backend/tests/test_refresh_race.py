"""真实任务调度器回归测试（P2 刷新竞态）。

走 _get_or_start_live_fetch 的真实入口：采集途中更新配置 → 旧结果不入库 →
任务退出后按最新配置自动重刷；连续编辑只使用最新配置。
"""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class RefreshRaceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="openusage-refrace-test-")
        self.original_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self.temp_dir.name, "test.db")
        await db.init_db()
        self.user_id = await db.create_user("race-user", "test-password")
        self.service_id = await db.add_user_service(
            self.user_id, "custom", "Private API", {"api_key": "secret-v1"}
        )
        self.release = asyncio.Event()
        self.calls = []

        async def fake_fetcher(config, *a, **kw):
            self.calls.append(dict(config))
            if len(self.calls) == 1:
                await self.release.wait()  # 第一次采集阻塞，模拟回源耗时
            return {
                "configured": True,
                "status": "ok",
                "data": {"fields": [{
                    "path": "data.quota", "label": "余额", "symbol": "$",
                    "type": "balance", "value": 100 - len(self.calls) * 10,
                    "history": True,
                }]},
            }

        self._fetchers_patch = patch.dict(
            backend_main.SERVICE_FETCHERS, {"custom": fake_fetcher}
        )
        self._fetchers_patch.start()
        self.addCleanup(self._fetchers_patch.stop)

    async def asyncTearDown(self):
        db.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    async def _wait_for_history(self, count: int, timeout_s: float = 5.0) -> bool:
        for _ in range(int(timeout_s / 0.05)):
            history = await db.get_service_usage_history(self.service_id, None, None)
            if len(history) == count:
                return True
            await asyncio.sleep(0.05)
        return False

    async def test_midflight_update_discards_and_refetches_with_new_config(self):
        svc_v1 = dict(await db.get_user_service(self.service_id, self.user_id))
        backend_main._get_or_start_live_fetch(svc_v1)  # 真实调度入口
        await asyncio.sleep(0.1)  # 第一次采集阻塞中
        self.assertEqual(len(self.calls), 1)

        # 采集途中更新配置（v2：改名 + 换凭据）
        await db.update_user_service(self.service_id, self.user_id, "Renamed", {"api_key": "new"})

        self.release.set()
        self.assertTrue(await self._wait_for_history(1), "新配置的采集未在限时内入库")

        svc = await db.get_user_service(self.service_id, self.user_id)
        self.assertEqual(svc["name"], "Renamed")              # 用户改名未被旧任务覆盖
        self.assertEqual(svc["config"], {"api_key": "new"})   # 新凭据未被覆盖
        self.assertEqual(svc["config_version"], 2)
        # 旧结果（value=90）未入库：唯一历史条目来自新配置的采集（value=80）
        history = await db.get_service_usage_history(self.service_id, None, None)
        self.assertEqual([e["data"]["fields"][0]["value"] for e in history], [80])
        # 第一次采集用旧凭据，重刷用的是新凭据
        self.assertEqual(self.calls[0]["api_key"], "secret-v1")
        self.assertEqual(self.calls[1]["api_key"], "new")

    async def test_consecutive_edits_use_latest_config(self):
        svc_v1 = dict(await db.get_user_service(self.service_id, self.user_id))
        backend_main._get_or_start_live_fetch(svc_v1)
        await asyncio.sleep(0.1)
        # 阻塞期间连续两次编辑
        await db.update_user_service(self.service_id, self.user_id, "Rename-A", {"api_key": "key-a"})
        await db.update_user_service(self.service_id, self.user_id, "Rename-B", {"api_key": "key-b"})
        self.release.set()
        self.assertTrue(await self._wait_for_history(1))

        history = await db.get_service_usage_history(self.service_id, None, None)
        self.assertEqual([e["data"]["fields"][0]["value"] for e in history], [80])
        svc = await db.get_user_service(self.service_id, self.user_id)
        self.assertEqual(svc["name"], "Rename-B")             # 只使用最新配置
        self.assertEqual(svc["config"], {"api_key": "key-b"})
        self.assertEqual(self.calls[-1]["api_key"], "key-b")
        self.assertEqual(len(self.calls), 2)                  # 没有针对中间态的多余采集


if __name__ == "__main__":
    unittest.main()
