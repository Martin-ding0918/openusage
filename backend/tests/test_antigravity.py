import os
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

import httpx

from fetchers import antigravity
from thresholds import APPLICABLE_THRESHOLDS, extract_history_metrics, extract_usage_percentages


def _json_response(url: str, status_code: int, payload: dict) -> httpx.Response:
    req = httpx.Request("POST", url)
    return httpx.Response(status_code, json=payload, request=req)


class AntigravityOAuthAndImportTests(unittest.TestCase):
    def setUp(self):
        antigravity._pending_oauth_flows.clear()

    def test_start_oauth_flow_builds_pkce_url(self):
        res = antigravity.start_oauth_flow(client_id="cid-123.apps.googleusercontent.com", client_secret="sec-1")
        self.assertEqual(res["status"], "ok")
        parsed = urlparse(res["auth_url"])
        qs = parse_qs(parsed.query)
        self.assertEqual(qs["client_id"], ["cid-123.apps.googleusercontent.com"])
        self.assertEqual(qs["code_challenge_method"], ["S256"])
        self.assertEqual(qs["access_type"], ["offline"])
        self.assertEqual(qs["state"], [res["state"]])
        self.assertIn(res["state"], antigravity._pending_oauth_flows)

    def test_start_oauth_flow_requires_client_id(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTIGRAVITY_OAUTH_CLIENT_ID", None)
            with self.assertRaises(ValueError):
                antigravity.start_oauth_flow()

    def test_parse_oauth_callback_variants(self):
        code, state = antigravity.parse_oauth_callback_input(
            "http://localhost:51121/oauth-callback?state=st1&code=4/0Abc123&scope=email"
        )
        self.assertEqual((code, state), ("4/0Abc123", "st1"))

        code2, state2 = antigravity.parse_oauth_callback_input("code=4/0Raw&state=st2")
        self.assertEqual((code2, state2), ("4/0Raw", "st2"))

        code3, state3 = antigravity.parse_oauth_callback_input("4/0DirectCode")
        self.assertEqual((code3, state3), ("4/0DirectCode", None))

        with self.assertRaises(ValueError):
            antigravity.parse_oauth_callback_input("http://localhost:51121/oauth-callback?error=access_denied")

    def test_parse_credentials_raw_and_json_formats(self):
        raw_refresh = antigravity.parse_antigravity_credentials("1//0gRefreshToken")
        self.assertEqual(raw_refresh["refresh_token"], "1//0gRefreshToken")
        self.assertEqual(raw_refresh["access_token"], "")

        raw_access = antigravity.parse_antigravity_credentials("Bearer ya29.AccessToken")
        self.assertEqual(raw_access["access_token"], "ya29.AccessToken")
        self.assertEqual(raw_access["refresh_token"], "")

        acp_json = antigravity.parse_antigravity_credentials(
            '{"client_id":"cid","client_secret":"csec","refresh_token":"1//acp","project_id":"proj-1"}'
        )
        self.assertEqual(acp_json["refresh_token"], "1//acp")
        self.assertEqual(acp_json["client_id"], "cid")
        self.assertEqual(acp_json["client_secret"], "csec")
        self.assertEqual(acp_json["project_id"], "proj-1")

        accounts_json = antigravity.parse_antigravity_credentials(
            '{"activeIndex":1,"accounts":[{"refreshToken":"1//first"},{"refreshToken":"1//second","projectId":"p2","email":"b@example.com"}]}'
        )
        self.assertEqual(accounts_json["refresh_token"], "1//second")
        self.assertEqual(accounts_json["project_id"], "p2")
        self.assertEqual(accounts_json["email"], "b@example.com")

        picoclaw_json = antigravity.parse_antigravity_credentials(
            '{"credentials":{"google-antigravity":{"refresh":"1//pico","access":"ya29.p","projectId":"p3"}}}'
        )
        self.assertEqual(picoclaw_json["refresh_token"], "1//pico")
        self.assertEqual(picoclaw_json["access_token"], "ya29.p")
        self.assertEqual(picoclaw_json["project_id"], "p3")

        with self.assertRaises(ValueError):
            antigravity.parse_antigravity_credentials('{"foo":"bar"}')


class AntigravityNormalizationTests(unittest.TestCase):
    def test_summary_groups_hide_untouched_reset_and_handle_proto3_zero(self):
        payload = {
            "groups": [
                {
                    "displayName": "Claude and GPT models",
                    "buckets": [
                        {
                            "bucketId": "3p-weekly",
                            "window": "weekly",
                            "remainingFraction": 0.49494,
                            "resetTime": "2026-11-05T13:10:02Z",
                        },
                        {
                            "bucketId": "3p-5h",
                            "window": "5h",
                            "remainingFraction": 1,
                            "resetTime": "2026-10-30T15:05:33Z",
                        },
                    ],
                },
                {
                    "displayName": "Gemini 4 Argon",
                    "buckets": [
                        {
                            # Proto3 omits remainingFraction when exhausted (0.0).
                            "bucketId": "gemini-argon-wtu-weekly",
                            "window": "weekly",
                            "resetTime": "2026-11-05T18:33:13Z",
                        }
                    ],
                },
                {
                    "displayName": "Gemini Models",
                    "buckets": [
                        {
                            "bucketId": "gemini-weekly",
                            "window": "weekly",
                            "remainingFraction": 0.85,
                            "resetTime": "2026-11-06T10:05:33Z",
                        },
                        {
                            "bucketId": "gemini-5h",
                            "window": "5h",
                            "remainingFraction": 0.7466,
                            "resetTime": "2026-10-30T14:00:00Z",
                        },
                    ],
                },
            ]
        }

        groups = antigravity._normalize_summary_payload(payload)
        self.assertEqual([g["id"] for g in groups], ["gemini", "claude_gpt", "argon"])

        # Within each group, 5h comes before weekly.
        self.assertEqual([b["id"] for b in groups[0]["buckets"]], ["gemini-5h", "gemini-weekly"])
        self.assertEqual(groups[0]["buckets"][0]["percent_used"], 25.3)
        self.assertEqual(groups[0]["buckets"][0]["percent_remaining"], 74.7)
        self.assertEqual(groups[0]["buckets"][0]["resets_at"], "2026-10-30T14:00:00Z")

        # Untouched 100% bucket hides synthetic rolling resetTime.
        claude_5h = groups[1]["buckets"][0]
        self.assertEqual(claude_5h["id"], "3p-5h")
        self.assertEqual(claude_5h["percent_used"], 0.0)
        self.assertIsNone(claude_5h["resets_at"])

        # Exhausted proto3 omitted remainingFraction -> 100% used, 0% remaining.
        argon = groups[2]["buckets"][0]
        self.assertEqual(argon["percent_used"], 100.0)
        self.assertEqual(argon["percent_remaining"], 0.0)
        self.assertEqual(argon["resets_at"], "2026-11-05T18:33:13Z")

    def test_thresholds_and_history_metrics(self):
        self.assertEqual(APPLICABLE_THRESHOLDS["antigravity"], ("session", "weekly"))
        data = {
            "limits": [
                {"id": "gemini-5h", "window": "5h", "label": "Gemini · 5h", "percent_used": 25.0},
                {"id": "gemini-weekly", "window": "weekly", "label": "Gemini · Weekly", "percent_used": 15.0},
                {"id": "3p-5h", "window": "5h", "label": "Claude & GPT · 5h", "percent_used": 60.0},
                {"id": "gemini-argon-wtu-weekly", "window": "weekly", "label": "Argon · Weekly", "percent_used": 100.0},
            ]
        }
        pcts = extract_usage_percentages("antigravity", data)
        self.assertEqual(pcts, {"session": 60.0, "weekly": 100.0})

        history = extract_history_metrics("antigravity", data)
        self.assertEqual(history["session"]["value"], 60.0)
        self.assertEqual(history["weekly"]["value"], 100.0)
        self.assertEqual(history["bucket:gemini-5h"]["label"], "Gemini · 5h")
        self.assertEqual(history["bucket:gemini-argon-wtu-weekly"]["value"], 100.0)


class AntigravityFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_unconfigured_returns_not_configured(self):
        res = await antigravity.fetch_antigravity({})
        self.assertFalse(res["configured"])

    async def test_refreshes_token_persists_and_fetches_daily_quota(self):
        saved_updates = []

        async def on_update(u: dict):
            saved_updates.append(u)

        async def fake_post(self_client, url, **kwargs):
            if url == antigravity.GOOGLE_TOKEN_URL:
                return _json_response(url, 200, {"access_token": "ya29.fresh", "expires_in": 3600})
            if url.endswith(antigravity.LOAD_CODE_ASSIST_PATH):
                return _json_response(
                    url,
                    200,
                    {"cloudaicompanionProject": "aicode-consumers", "currentTier": {"name": "Google AI Pro"}},
                )
            if url == f"{antigravity.DAILY_CLOUDCODE_BASE}{antigravity.QUOTA_SUMMARY_PATH}":
                self.assertIn("antigravity/hub/2.24.0", kwargs["headers"]["User-Agent"])
                return _json_response(
                    url,
                    200,
                    {
                        "groups": [
                            {
                                "displayName": "Gemini Models",
                                "buckets": [
                                    {"bucketId": "gemini-5h", "window": "5h", "remainingFraction": 0.8, "resetTime": "2026-11-26T20:00:00Z"},
                                    {"bucketId": "gemini-weekly", "window": "weekly", "remainingFraction": 0.9, "resetTime": "2026-12-01T00:00:00Z"},
                                ],
                            },
                            {
                                "displayName": "Claude and GPT models",
                                "buckets": [
                                    {"bucketId": "3p-5h", "window": "5h", "remainingFraction": 1.0, "resetTime": "2026-11-26T21:00:00Z"},
                                ],
                            },
                        ]
                    },
                )
            raise AssertionError(f"Unexpected POST to {url}")

        async def fake_get(self_client, url, **kwargs):
            if url == antigravity.GOOGLE_USERINFO_URL:
                return _json_response(url, 200, {"email": "dev@gmail.com"})
            raise AssertionError(f"Unexpected GET to {url}")

        with patch.object(httpx.AsyncClient, "post", new=fake_post), \
                patch.object(httpx.AsyncClient, "get", new=fake_get):
            result = await antigravity.fetch_antigravity(
                {
                    "antigravity_refresh_token": "1//refresh",
                    "antigravity_client_id": "cid",
                    "antigravity_client_secret": "csec",
                },
                on_config_update=on_update,
            )

        self.assertEqual(result["status"], "ok")
        data = result["data"]
        self.assertEqual(data["plan"], "Google AI Pro")
        self.assertEqual(data["email"], "dev@gmail.com")
        self.assertEqual(data["project_id"], "aicode-consumers")
        self.assertEqual(len(data["groups"]), 2)
        self.assertEqual(len(data["limits"]), 3)
        self.assertEqual(data["limits"][0]["percent_used"], 20.0)
        self.assertEqual(len(saved_updates), 1)
        self.assertEqual(saved_updates[0]["antigravity_access_token"], "ya29.fresh")
        self.assertEqual(saved_updates[0]["antigravity_email"], "dev@gmail.com")

    async def test_partial_daily_merges_missing_gemini_from_prod_without_clobbering_3p(self):
        async def fake_post(self_client, url, **kwargs):
            if url.endswith(antigravity.LOAD_CODE_ASSIST_PATH):
                return _json_response(url, 200, {"cloudaicompanionProject": "aicode-consumers"})
            if url == f"{antigravity.DAILY_CLOUDCODE_BASE}{antigravity.QUOTA_SUMMARY_PATH}":
                return _json_response(
                    url,
                    200,
                    {
                        "groups": [
                            {
                                "displayName": "Claude and GPT models",
                                "buckets": [
                                    {"bucketId": "3p-5h", "window": "5h", "remainingFraction": 0.4, "resetTime": "2026-11-26T20:00:00Z"},
                                ],
                            }
                        ]
                    },
                )
            if url == f"{antigravity.PROD_CLOUDCODE_BASE}{antigravity.QUOTA_SUMMARY_PATH}":
                return _json_response(
                    url,
                    200,
                    {
                        "groups": [
                            {
                                "displayName": "Gemini Models",
                                "buckets": [
                                    {"bucketId": "gemini-5h", "window": "5h", "remainingFraction": 0.7, "resetTime": "2026-11-26T20:00:00Z"},
                                ],
                            },
                            {
                                "displayName": "Claude and GPT models",
                                "buckets": [
                                    # Prod reports unused 1.0, must NOT overwrite daily's 0.4!
                                    {"bucketId": "3p-5h", "window": "5h", "remainingFraction": 1.0, "resetTime": "2026-11-26T21:00:00Z"},
                                ],
                            },
                        ]
                    },
                )
            raise AssertionError(f"Unexpected POST to {url}")

        with patch.object(httpx.AsyncClient, "post", new=fake_post):
            result = await antigravity.fetch_antigravity(
                {
                    "antigravity_access_token": "ya29.valid",
                    "antigravity_token_expires_at": "2099-01-01T00:00:00Z",
                    "antigravity_email": "dev@gmail.com",
                }
            )

        self.assertEqual(result["status"], "ok")
        by_id = {b["id"]: b for b in result["data"]["limits"]}
        self.assertEqual(by_id["gemini-5h"]["percent_remaining"], 70.0)
        self.assertEqual(by_id["3p-5h"]["percent_remaining"], 40.0)

    async def test_401_on_cached_token_triggers_refresh_and_retry(self):
        calls = []

        async def fake_post(self_client, url, **kwargs):
            calls.append((url, kwargs.get("headers", {}).get("Authorization")))
            if url == antigravity.GOOGLE_TOKEN_URL:
                return _json_response(url, 200, {"access_token": "ya29.new", "expires_in": 3600})
            if kwargs.get("headers", {}).get("Authorization") == "Bearer ya29.stale":
                return _json_response(url, 401, {"error": {"status": "UNAUTHENTICATED", "message": "expired"}})
            if url.endswith(antigravity.LOAD_CODE_ASSIST_PATH):
                return _json_response(url, 200, {"cloudaicompanionProject": "aicode-consumers"})
            if url.endswith(antigravity.QUOTA_SUMMARY_PATH):
                return _json_response(
                    url,
                    200,
                    {
                        "groups": [
                            {
                                "displayName": "Gemini Models",
                                "buckets": [
                                    {"bucketId": "gemini-5h", "window": "5h", "remainingFraction": 0.9, "resetTime": "2026-11-26T20:00:00Z"}
                                ],
                            }
                        ]
                    },
                )
            raise AssertionError(f"Unexpected POST to {url}")

        with patch.object(httpx.AsyncClient, "post", new=fake_post):
            result = await antigravity.fetch_antigravity(
                {
                    "antigravity_access_token": "ya29.stale",
                    "antigravity_token_expires_at": "2099-01-01T00:00:00Z",
                    "antigravity_refresh_token": "1//refresh",
                    "antigravity_client_id": "cid",
                    "antigravity_email": "dev@gmail.com",
                }
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["limits"][0]["percent_remaining"], 90.0)
        self.assertTrue(any(url == antigravity.GOOGLE_TOKEN_URL for url, _ in calls))

    async def test_fallback_to_fetch_available_models_when_summary_empty(self):
        async def fake_post(self_client, url, **kwargs):
            if url.endswith(antigravity.LOAD_CODE_ASSIST_PATH):
                return _json_response(url, 200, {"cloudaicompanionProject": "aicode-consumers"})
            if url.endswith(antigravity.QUOTA_SUMMARY_PATH):
                return _json_response(url, 200, {"groups": []})
            if url.endswith(antigravity.FETCH_MODELS_PATH):
                return _json_response(
                    url,
                    200,
                    {
                        "models": {
                            "gemini-3-pro": {"quotaInfo": {"remainingFraction": 0.65, "resetTime": "2026-11-26T20:00:00Z"}},
                            "claude-sonnet-4-5": {"quotaInfo": {"remainingFraction": 0.80, "resetTime": "2026-11-26T20:00:00Z"}},
                            "gemini-4-argon-high": {"quotaInfo": {"remainingFraction": 0.50, "resetTime": "2026-12-01T00:00:00Z"}},
                        }
                    },
                )
            raise AssertionError(f"Unexpected POST to {url}")

        with patch.object(httpx.AsyncClient, "post", new=fake_post):
            result = await antigravity.fetch_antigravity(
                {
                    "antigravity_access_token": "ya29.valid",
                    "antigravity_token_expires_at": "2099-01-01T00:00:00Z",
                    "antigravity_email": "dev@gmail.com",
                }
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual([g["id"] for g in result["data"]["groups"]], ["gemini", "claude_gpt", "argon"])
        self.assertEqual(result["data"]["limits"][0]["percent_remaining"], 65.0)

    async def test_exchange_oauth_code_state_mismatch_raises(self):
        antigravity._pending_oauth_flows.clear()
        started = antigravity.start_oauth_flow(client_id="cid")
        with self.assertRaises(ValueError):
            await antigravity.exchange_oauth_code(
                "http://localhost:51121/oauth-callback?code=abc&state=wrong",
                state=started["state"],
            )


if __name__ == "__main__":
    unittest.main()
