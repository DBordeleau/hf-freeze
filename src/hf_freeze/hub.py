"""Metadata-only Hugging Face Hub revision resolution."""

from __future__ import annotations

from typing import Protocol

from huggingface_hub import HfApi
from huggingface_hub.errors import (
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from hf_freeze.models import RepoType


class HubResolutionError(Exception):
    """A safe, user-facing failure to resolve a Hub revision."""


class HubResolver(Protocol):
    """The only Hub operation needed to create schema-v1 lockfiles."""

    def resolve(self, repo_id: str, repo_type: RepoType, revision: str) -> str:
        """Return the exact commit SHA for a repository revision."""
        ...


class HfHubResolver:
    """Resolve repository revisions through public Hub metadata APIs."""

    def __init__(self, api: HfApi | None = None) -> None:
        self._api = api or HfApi()

    def resolve(self, repo_id: str, repo_type: RepoType, revision: str) -> str:
        try:
            info = self._api.repo_info(
                repo_id=repo_id,
                repo_type=repo_type.value,
                revision=revision,
            )
        except RevisionNotFoundError as error:
            raise HubResolutionError(
                f"revision '{revision}' was not found for {repo_type.value} "
                f"repository '{repo_id}'"
            ) from error
        except RepositoryNotFoundError as error:
            raise HubResolutionError(
                f"{repo_type.value} repository '{repo_id}' was not found or is not "
                "accessible"
            ) from error
        except HfHubHTTPError as error:
            raise HubResolutionError(
                f"Hub request failed while resolving {repo_type.value} repository "
                f"'{repo_id}' at revision '{revision}'"
            ) from error

        sha = info.sha
        if not isinstance(sha, str) or not sha:
            raise HubResolutionError(
                f"Hub returned no commit SHA for {repo_type.value} repository "
                f"'{repo_id}' at revision '{revision}'"
            )
        return sha
