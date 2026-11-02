"""Port of src/lib/azure-openai.ts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

from ._js import Record, decode_uri_component, encode_uri_component, js_trim, rx

AZURE_OPENAI_API_VERSION = "2024-10-21"

_SCHEME_RE = re.compile(r"^https?://", re.I)
_FALLBACK_HOST_RE = rx(r"(^|//)[^/?#]+\.openai\.azure\.com(?::\d+)?(?:[/?#]|$)".replace("$", r"\Z"), re.I)
_API_VERSION_RE = rx(r"[?&]api-version=([^&]+)", re.I)
_WITH_DEPLOYMENT_RE = rx(
    r"\A(https?://[^/]+\.openai\.azure\.com)/openai/deployments/([^/]+)(?:/chat/completions)?\Z", re.I
)
_RESOURCE_ONLY_RE = rx(r"\A(https?://[^/]+\.openai\.azure\.com)(?:/openai)?\Z", re.I)
_TRAILING_SLASHES_RE = re.compile(r"/+\Z")


def is_azure_openai_endpoint(endpoint: str) -> bool:
    trimmed = js_trim(endpoint)
    if not trimmed:
        return False
    try:
        url = urlsplit(trimmed if _SCHEME_RE.search(trimmed) else f"https://{trimmed}")
        hostname = url.hostname
        if hostname is None:
            raise ValueError("no host")
        return hostname.lower().endswith(".openai.azure.com")
    except ValueError:
        return bool(_FALLBACK_HOST_RE.search(trimmed))


@dataclass
class AzureParsedEndpoint(Record):
    resourceBase: str
    deployment: str
    apiVersion: str


def parse_azure_openai_endpoint(
    endpoint: str, fallback_deployment: str, fallback_api_version: str
) -> Optional[AzureParsedEndpoint]:
    trimmed = js_trim(endpoint)
    if not is_azure_openai_endpoint(trimmed):
        return None

    api_version = js_trim(fallback_api_version) or AZURE_OPENAI_API_VERSION
    q_match = _API_VERSION_RE.search(trimmed)
    if q_match:
        api_version = decode_uri_component(q_match.group(1))

    without_query = _TRAILING_SLASHES_RE.sub("", trimmed.split("?")[0])

    with_deployment = _WITH_DEPLOYMENT_RE.search(without_query)
    if with_deployment:
        return AzureParsedEndpoint(
            resourceBase=with_deployment.group(1),
            deployment=decode_uri_component(with_deployment.group(2)),
            apiVersion=api_version,
        )

    resource_only = _RESOURCE_ONLY_RE.search(without_query)
    if resource_only:
        deployment = js_trim(fallback_deployment)
        if not deployment:
            return None
        return AzureParsedEndpoint(resourceBase=resource_only.group(1), deployment=deployment, apiVersion=api_version)

    return None


def build_azure_openai_url(endpoint: str, deployment: str, api_version: str) -> str:
    parsed = parse_azure_openai_endpoint(endpoint, deployment, api_version)
    if not parsed:
        trimmed = _TRAILING_SLASHES_RE.sub("", endpoint)
        version = encode_uri_component(js_trim(api_version) or AZURE_OPENAI_API_VERSION)
        deployment_path = f"/openai/deployments/{encode_uri_component(deployment)}/chat/completions"
        return f"{trimmed}{deployment_path}?api-version={version}"
    version = encode_uri_component(parsed.apiVersion)
    dep = encode_uri_component(parsed.deployment)
    return f"{parsed.resourceBase}/openai/deployments/{dep}/chat/completions?api-version={version}"
