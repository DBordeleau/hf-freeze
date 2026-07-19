"""Narrow public-API boundary for Hugging Face Hub metadata."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Protocol

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.hf_api import RepoFile

from hf_freeze.diff import SEMANTIC_FILES, SEMANTIC_SIZE_LIMIT, RepositoryFile
from hf_freeze.models import RepoType


class HubResolutionError(Exception):
    """A safe, user-facing failure to resolve a Hub revision."""


class HubTreeError(HubResolutionError):
    """A safe, user-facing failure to retrieve repository metadata."""


class HubContentError(HubResolutionError):
    """An allowlisted small file could not be read for semantic comparison."""


class HubResolver(Protocol):
    """The only Hub operation needed to create schema-v1 lockfiles."""

    def resolve(self, repo_id: str, repo_type: RepoType, revision: str) -> str:
        """Return the exact commit SHA for a repository revision."""
        ...


class HfHubResolver:
    """Resolve repository revisions through public Hub metadata APIs."""

    def __init__(
        self,
        api: HfApi | None = None,
        downloader: Callable[..., str] = hf_hub_download,
    ) -> None:
        self._api = api or HfApi()
        self._downloader = downloader

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

    def tree(
        self, repo_id: str, repo_type: RepoType, revision: str
    ) -> tuple[RepositoryFile, ...]:
        """Return a recursive file tree with expanded content identities."""

        try:
            entries = tuple(
                self._api.list_repo_tree(
                    repo_id=repo_id,
                    repo_type=repo_type.value,
                    revision=revision,
                    recursive=True,
                    expand=True,
                )
            )
        except RevisionNotFoundError as error:
            raise HubTreeError(
                f"revision '{revision}' was not found while reading repository "
                f"tree for '{repo_id}'"
            ) from error
        except RepositoryNotFoundError as error:
            raise HubTreeError(
                f"{repo_type.value} repository '{repo_id}' was not found or is not "
                "accessible"
            ) from error
        except HfHubHTTPError as error:
            raise HubTreeError(
                f"Hub request failed while reading repository tree for '{repo_id}' "
                f"at revision '{revision}'"
            ) from error

        return tuple(
            RepositoryFile(
                path=entry.path,
                size=entry.size if isinstance(entry.size, int) else None,
                lfs_sha256=entry.lfs.sha256 if entry.lfs is not None else None,
                xet_hash=entry.xet_hash,
                blob_id=entry.blob_id or None,
            )
            for entry in entries
            if isinstance(entry, RepoFile)
        )

    def read_small_file(
        self,
        repo_id: str,
        repo_type: RepoType,
        revision: str,
        path: str,
        expected_size: int | None,
    ) -> str:
        """Download one CLI-allowlisted small JSON file and return UTF-8 text."""

        if PurePosixPath(path).name not in SEMANTIC_FILES:
            raise HubContentError(f"semantic download is not allowed for '{path}'")
        if (
            type(expected_size) is not int
            or expected_size < 0
            or expected_size > SEMANTIC_SIZE_LIMIT
        ):
            raise HubContentError(
                f"semantic file '{path}' has an unsafe or unavailable metadata size"
            )
        try:
            local_path = self._downloader(
                repo_id=repo_id,
                repo_type=repo_type.value,
                revision=revision,
                filename=path,
            )
            content = Path(local_path).read_bytes()
        except (EntryNotFoundError, RevisionNotFoundError) as error:
            raise HubContentError(
                f"semantic file '{path}' was unavailable at revision '{revision}'"
            ) from error
        except (RepositoryNotFoundError, HfHubHTTPError, OSError) as error:
            raise HubContentError(
                f"Hub request failed while reading semantic file '{path}'"
            ) from error
        if len(content) > SEMANTIC_SIZE_LIMIT:
            raise HubContentError(f"semantic file '{path}' exceeded 64 KiB")
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise HubContentError(
                f"semantic file '{path}' is not valid UTF-8"
            ) from error
