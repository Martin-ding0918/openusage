"""历史区间完整覆盖测试（P2「历史区间被静默截断」）。

250 条 5 分钟粒度样本（跨约 20.8 小时）：汇总基于全量样本、图表点按 max_points
降采样且始终保留最新点、coverage 如实反映范围与条数、limit 截断不再静默。
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
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

TOTAL = 250
STEP_MINUTES = 5
START = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)


def _entry(value: float, ts: str) -> dict:
    return {
        "fetched_at": ts,
        "data": {"fields": [{
            "path": "data.quota", "label": "余额", "symbol": "$",
            "type": "balance", "value": value, "history": True,
        }]},
    }


class HistoryFullRangeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="openusage-histfull-test-")
        self.original_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self.temp_dir.name, "test.db")
        await db.init_db()
        self.user_id = await db.create_user("hist-user", "test-password")
        self.service_id = await db.add_user_service(
            self.user_id, "custom", "Private API", {"api_key": "secret"}, store_history=True
        )
        # 250 条样本：value 从 100.0 线性降到 75.1（步长 0.1），时间 5 分钟递增
        for i in range(TOTAL):
            value = round(100.0 - i * 0.1, 2)
            await db.save_service_usage(self.service_id, {"fields": [{
                "path": "data.quota", "label": "余额", "symbol": "$",
                "type": "balance", "value": value, "history": True,
            }]})
        # save_service_usage 固定 fetched_at=now：逐条改写为递增时间戳
        import aiosqlite
        async with aiosqlite.connect(db.DB_PATH) as db_conn:
            async with db_conn.execute(
                "SELECT id FROM service_usage_history WHERE user_service_id = ? ORDER BY id",
                (self.service_id,),
            ) as cursor:
                ids = [row[0] for row in await cursor.fetchall()]
            self.assertEqual(len(ids), TOTAL)
            for i, row_id in enumerate(ids):
                ts = (START + timedelta(minutes=STEP_MINUTES * i)).isoformat()
                await db_conn.execute(
                    "UPDATE service_usage_history SET fetched_at = ? WHERE id = ?", (ts, row_id)
                )
            await db_conn.commit()

    async def asyncTearDown(self):
        db.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    async def test_full_range_summary_and_downsampling(self):
        with patch.object(backend_main, "get_user_service", AsyncMock(return_value={
            "service_type": "custom", "store_history": True,
        })):
            response = await backend_main.get_service_history(
                self.service_id, since=None, limit=10000, max_points=20, user={"id": self.user_id}
            )
        self.assertFalse(response["coverage"]["truncated"])
        self.assertEqual(response["coverage"]["total_entries"], TOTAL)
        series = response["series"]["field:data.quota"]
        # 全样本汇总
        self.assertEqual(series["summary"]["count"], TOTAL)
        self.assertAlmostEqual(series["summary"]["avg"], 87.55, places=2)
        self.assertAlmostEqual(series["summary"]["min"], 75.1, places=2)
        self.assertAlmostEqual(series["summary"]["max"], 100.0, places=2)
        # 图表点降采样到 max_points，且始终保留最新点
        self.assertLessEqual(len(series["points"]), 20)
        last = series["points"][-1]
        self.assertAlmostEqual(last["v"], 75.1, places=2)
        # 降采样点保持时间正序
        ts_list = [p["t"] for p in series["points"]]
        self.assertEqual(ts_list, sorted(ts_list))

    async def test_limit_truncation_is_visible(self):
        with patch.object(backend_main, "get_user_service", AsyncMock(return_value={
            "service_type": "custom", "store_history": True,
        })):
            response = await backend_main.get_service_history(
                self.service_id, since=None, limit=100, max_points=400, user={"id": self.user_id}
            )
        cov = response["coverage"]
        self.assertTrue(cov["truncated"])
        self.assertEqual(cov["total_entries"], TOTAL)
        # 只返回最新 100 条：第一条应是第 151 个样本（时间索引 150）
        first_expected = START + timedelta(minutes=STEP_MINUTES * (TOTAL - 100))
        self.assertEqual(cov["first"], first_expected.isoformat())
        self.assertEqual(cov["last"], (START + timedelta(minutes=STEP_MINUTES * (TOTAL - 1))).isoformat())
        series = response["series"]["field:data.quota"]
        self.assertEqual(series["summary"]["count"], 100)  # 汇总基于实际返回样本


if __name__ == "__main__":
    unittest.main()
