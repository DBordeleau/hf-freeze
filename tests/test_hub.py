from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import (
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from hf_freeze.hub import HfHubResolver, HubResolutionError
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
