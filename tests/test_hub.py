from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import (
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.hf_api import RepoFile

from hf_freeze.diff import SEMANTIC_SIZE_LIMIT, RepositoryFile
from hf_freeze.hub import (
    HfHubResolver,
    HubContentError,
    HubResolutionError,
    HubTreeError,
)
from hf_freeze.models import RepoType


def hub_error(error_type: type[HfHubHTTPError]) -> HfHubHTTPError:
    response = httpx.Response(
        404, request=httpx.Request("GET", "https://huggingface.co/api/models/repo")
    )
    return error_type("secret-token", response=response)


class FakeApi:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def repo_info(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeTreeApi(FakeApi):
    def list_repo_tree(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.parametrize("repo_type", [RepoType.MODEL, RepoType.DATASET])
def test_resolver_maps_repo_types_and_returns_sha(repo_type: RepoType) -> None:
    api = FakeApi(SimpleNamespace(sha="exact-sha"))

    assert HfHubResolver(api).resolve("org/repo", repo_type, "v1") == "exact-sha"
    assert api.calls == [
        {"repo_id": "org/repo", "repo_type": repo_type.value, "revision": "v1"}
    ]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (hub_error(RepositoryNotFoundError), "not found or is not accessible"),
        (hub_error(RevisionNotFoundError), "revision 'v1' was not found"),
        (hub_error(HfHubHTTPError), "Hub request failed"),
    ],
)
def test_resolver_translates_expected_errors_without_leaking_details(
    error: Exception, message: str
) -> None:
    with pytest.raises(HubResolutionError, match=message) as caught:
        HfHubResolver(FakeApi(error)).resolve("org/repo", RepoType.MODEL, "v1")

    assert "secret-token" not in str(caught.value)


def test_tree_requests_recursive_expanded_metadata_and_maps_identities() -> None:
    entry = RepoFile(
        path="model.safetensors",
        size=12,
        oid="blob",
        lfs={"size": 12, "oid": "lfs", "pointerSize": 100},
        xetHash="xet",
    )
    api = FakeTreeApi([entry])

    assert HfHubResolver(api).tree("org/repo", RepoType.MODEL, "sha") == (
        RepositoryFile("model.safetensors", 12, "lfs", "xet", "blob"),
    )
    assert api.calls == [
        {
            "repo_id": "org/repo",
            "repo_type": "model",
            "revision": "sha",
            "recursive": True,
            "expand": True,
        }
    ]


def test_tree_translates_hub_failure_without_leaking_details() -> None:
    with pytest.raises(HubTreeError, match="reading repository tree") as caught:
        HfHubResolver(FakeTreeApi(hub_error(HfHubHTTPError))).tree(
            "org/repo", RepoType.MODEL, "sha"
        )

    assert "secret-token" not in str(caught.value)


def test_small_file_reader_uses_exact_revision(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"ok": true}', encoding="utf-8")
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(path)

    text = HfHubResolver(FakeApi(None), download).read_small_file(
        "org/repo", RepoType.MODEL, "sha", "config.json", 12
    )

    assert text == '{"ok": true}'
    assert calls == [
        {
            "repo_id": "org/repo",
            "repo_type": "model",
            "revision": "sha",
            "filename": "config.json",
        }
    ]


@pytest.mark.parametrize(
    ("path", "size"),
    [
        ("model.safetensors", 10),
        ("data.parquet", 10),
        ("config.json", None),
        ("config.json", -1),
        ("config.json", SEMANTIC_SIZE_LIMIT + 1),
    ],
)
def test_small_file_reader_rejects_unsafe_metadata_before_download(
    path: str, size: int | None
) -> None:
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        return "unused"

    with pytest.raises(HubContentError):
        HfHubResolver(FakeApi(None), download).read_small_file(
            "org/repo", RepoType.MODEL, "sha", path, size
        )

    assert calls == []
