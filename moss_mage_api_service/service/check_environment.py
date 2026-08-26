#!/usr/bin/env python3
"""Validate the isolated runtime before loading a checkpoint."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path

from service.backends import install_mamba_transformers_compat


EXPECTED = {
    "moss": {
        "torch": "2.8.0",
        "torchvision": "0.23.0",
        "transformers": "4.57.1",
        "huggingface-hub": "0.36.0",
        "tokenizers": "0.22.2",
        "accelerate": "1.13.0",
    },
    "mage": {
        "torch": "2.8.0",
        "torchvision": "0.23.0",
        "transformers": "5.7.0",
        "huggingface-hub": "1.5.0",
        "tokenizers": "0.22.2",
        "accelerate": "1.13.0",
        "mamba-ssm": "2.2.6.post3",
    },
}
IMPORT_NAMES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "huggingface-hub": "huggingface_hub",
    "tokenizers": "tokenizers",
    "accelerate": "accelerate",
    "mamba-ssm": "mamba_ssm",
}


def validate_versions(profile: str, installed: dict[str, str]) -> list[str]:
    return [
        f"{name}=={wanted} is required, found {installed.get(name, 'missing')}"
        for name, wanted in EXPECTED[profile].items()
        if installed.get(name) != wanted
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(EXPECTED), required=True)
    args = parser.parse_args()

    if args.profile == "mage":
        install_mamba_transformers_compat()
    installed: dict[str, str] = {}
    locations: dict[str, Path] = {}
    for distribution in EXPECTED[args.profile]:
        try:
            installed[distribution] = importlib.metadata.version(distribution)
            module = importlib.import_module(IMPORT_NAMES[distribution])
            locations[distribution] = Path(module.__file__).resolve()
        except Exception as exc:
            raise RuntimeError(f"Cannot import {distribution}: {exc}") from exc

    errors = validate_versions(args.profile, installed)
    trusted_roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    for distribution, location in locations.items():
        if not any(root == location or root in location.parents for root in trusted_roots):
            errors.append(f"{distribution} was imported from unexpected path {location}")
    if errors:
        raise RuntimeError(
            "; ".join(errors)
            + f". Repair with: bash scripts/setup_envs.sh {args.profile}"
        )

    summary = ", ".join(
        f"{name}={installed[name]} ({locations[name]})"
        for name in EXPECTED[args.profile]
    )
    print(f"[{args.profile} environment] {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
