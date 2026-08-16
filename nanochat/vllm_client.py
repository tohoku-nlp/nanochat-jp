"""Shared asynchronous HTTP client for vLLM evaluation endpoints."""

import asyncio
import json

import aiohttp


class VLLMError(RuntimeError):
    """Raised when vLLM cannot provide a valid evaluation response."""


class VLLMClient:
    """Manage authenticated, bounded, retrying requests to one vLLM server."""

    def __init__(
        self,
        base_url,
        model,
        api_key=None,
        concurrency=16,
        timeout=300.0,
        retry_limit=3,
    ):
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if retry_limit < 0:
            raise ValueError("retry_limit must be non-negative")
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.concurrency = concurrency
        self.timeout = timeout
        self.retry_limit = retry_limit
        self._session = None
        self._request_semaphore = asyncio.Semaphore(concurrency)

    async def __aenter__(self):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        connector = aiohttp.TCPConnector(
            limit=self.concurrency,
            limit_per_host=self.concurrency,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=headers,
        )
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _require_session(self):
        if self._session is None:
            raise RuntimeError(
                f"{type(self).__name__} must be used as an async context manager"
            )
        return self._session

    def _redact_api_key(self, text):
        if self.api_key:
            return text.replace(self.api_key, '<redacted>')
        return text

    async def _request_json(self, method, path, payload=None):
        """Request JSON, retrying only transport errors, 429, and 5xx."""
        session = self._require_session()
        url = f"{self.base_url}{path}"
        last_error = None

        for attempt in range(self.retry_limit + 1):
            status = None
            response_text = None
            try:
                async with self._request_semaphore:
                    async with session.request(method, url, json=payload) as response:
                        status = response.status
                        response_text = await response.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                detail = self._redact_api_key(str(exc))
                last_error = VLLMError(f"vLLM request failed for {path}: {detail}")
            else:
                if 200 <= status < 300:
                    try:
                        return json.loads(response_text)
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise VLLMError(
                            f"vLLM returned invalid JSON for {path}"
                        ) from exc

                message = response_text[:1000] if response_text else '<empty response>'
                message = self._redact_api_key(message)
                last_error = VLLMError(
                    f"vLLM returned HTTP {status} for {path}: {message}"
                )
                if status != 429 and not 500 <= status < 600:
                    raise last_error

            if attempt == self.retry_limit:
                assert last_error is not None
                raise last_error
            await asyncio.sleep(2 ** attempt)

        raise AssertionError("unreachable")

    async def validate_model(self):
        """Check connectivity and verify the requested served model name."""
        response = await self._request_json('GET', '/v1/models')
        models = response.get('data') if isinstance(response, dict) else None
        if not isinstance(models, list):
            raise VLLMError("/v1/models response is missing the data list")
        model_ids = [entry.get('id') for entry in models if isinstance(entry, dict)]
        if self.model not in model_ids:
            available = ', '.join(str(model_id) for model_id in model_ids) or '<none>'
            raise VLLMError(
                f"model {self.model!r} is not served; available models: {available}"
            )
