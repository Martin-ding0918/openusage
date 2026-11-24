import unittest
from unittest.mock import AsyncMock, patch

import main


class HistorySeriesTests(unittest.IsolatedAsyncioTestCase):
    async def test_renamed_field_keeps_points_and_latest_metadata(self):
        def entry(label, symbol, value, timestamp):
            return {"fetched_at": timestamp, "data": {"fields": [{
                "path": "data.quota", "label": label, "symbol": symbol,
                "type": "balance", "value": value, "history": True,
            }]}}
        history = [entry("旧余额", "", 46, "2026-11-15T01:00:00Z"),
                   entry("余额", "$", 45, "2026-11-15T02:00:00Z")]
        with patch.object(main, "get_user_service", AsyncMock(return_value={
            "service_type": "custom", "store_history": True,
        })), patch.object(main, "get_service_usage_history", AsyncMock(return_value=history)), patch.object(main, "count_service_usage_history", AsyncMock(return_value=len(history))):
            response = await main.get_service_history(3, limit=1000, max_points=400, user={"id": 1})
        self.assertEqual(set(response["series"]), {"field:data.quota"})
        series = response["series"]["field:data.quota"]
        self.assertEqual((series["label"], series["unit"]), ("余额", "$"))
        self.assertEqual([point["v"] for point in series["points"]], [46, 45])
