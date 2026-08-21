"""
Generate a CycloneDX Software Bill of Materials (SBOM) for youtube-to-docs.

The component list and dependency graph come from `uv export --format cyclonedx1.5`,
which reads `uv.lock` and therefore covers every extra (aws, azure, gcp, workspace,
m365, huggingface, audio, video, app) and every dependency group (dev, test).

The export is then enriched with information uv does not emit:
  * license declarations, read from the installed distribution metadata
  * artifact download URLs and SHA-256 hashes, read from `uv.lock`
  * root component metadata (description, license, purl, project links)

For complete license coverage the environment must be synced first:

    uv sync --all-extras --all-groups && uv run scripts/generate_sbom.py

Packages that are not installed (platform-specific wheels such as pywin32) are
reported at the end and are left without a license declaration rather than guessed.
"""

import json
import subprocess
import sys
import tomllib
from importlib.metadata import Distribution, PackageNotFoundError
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "sbom.cdx.json"

REPO_URL = "https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs"
DOCS_URL = "https://DoIT-Artificial-Intelligence.github.io/youtube-to-docs/"
PYPI_URL = "https://pypi.org/project/youtube-to-docs/"

# Trove classifiers that map unambiguously onto an SPDX license identifier.
# Ambiguous ones (plain "BSD License", bare "GNU General Public License") are
# recorded as free-text names instead of being resolved to a specific version.
CLASSIFIER_TO_SPDX = {
    "MIT License": "MIT",
    "MIT No Attribution License (MIT-0)": "MIT-0",
    "Apache Software License": "Apache-2.0",
    "BSD 3-Clause \"New\" or \"Revised\" License (BSD-3-Clause)": "BSD-3-Clause",
    "BSD 2-Clause \"Simplified\" License (BSD-2-Clause)": "BSD-2-Clause",
    "ISC License (ISCL)": "ISC",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "Python Software Foundation License": "PSF-2.0",
    "The Unlicense (Unlicense)": "Unlicense",
    "Zope Public License": "ZPL-2.1",
    "GNU Lesser General Public License v2 (LGPLv2)": "LGPL-2.0-only",
    "GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.0-or-later",
    "GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0-only",
    "GNU Lesser General Public License v3 or later (LGPLv3+)": "LGPL-3.0-or-later",
    "GNU General Public License v2 (GPLv2)": "GPL-2.0-only",
    "GNU General Public License v2 or later (GPLv2+)": "GPL-2.0-or-later",
    "GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
    "GNU General Public License v3 or later (GPLv3+)": "GPL-3.0-or-later",
    "Apache License 2.0": "Apache-2.0",
}

# Short values seen in the legacy `License:` metadata field.
LICENSE_FIELD_TO_SPDX = {
    "mit": "MIT",
    "mit license": "MIT",
    "apache 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "asl 2.0": "Apache-2.0",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause",
    "isc": "ISC",
    "mpl-2.0": "MPL-2.0",
    "psf-2.0": "PSF-2.0",
    "unlicense": "Unlicense",
    "0bsd": "0BSD",
}


def canonical(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return name.lower().replace("_", "-").replace(".", "-")


def run_uv_export() -> dict[str, Any]:
    """Export uv.lock as a CycloneDX 1.5 document covering all extras and groups."""
    result = subprocess.run(
        [
            "uv",
            "export",
            "--format",
            "cyclonedx1.5",
            "--all-extras",
            "--all-groups",
            "--frozen",
            "--quiet",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"uv export failed with exit code {result.returncode}")
    return json.loads(result.stdout)


def load_lock_artifacts() -> dict[str, list[dict[str, Any]]]:
    """Map each locked package to its sdist and wheel artifacts from uv.lock."""
    with open(REPO_ROOT / "uv.lock", "rb") as f:
        lock = tomllib.load(f)

    artifacts: dict[str, list[dict[str, Any]]] = {}
    for package in lock.get("package", []):
        entries = []
        sdist = package.get("sdist")
        if isinstance(sdist, dict) and sdist.get("url"):
            entries.append(sdist)
        entries.extend(w for w in package.get("wheels", []) if w.get("url"))
        if entries:
            artifacts[canonical(package["name"])] = entries
    return artifacts


def installed_distributions() -> dict[str, Distribution]:
    """Index the distributions installed in the current environment by name."""
    distributions: dict[str, Distribution] = {}
    for dist in Distribution.discover():
        name = dist.metadata["Name"]
        if name:
            distributions[canonical(name)] = dist
    return distributions


def spdx_or_name(value: str) -> dict[str, Any]:
    """Wrap a license string as an SPDX id when recognized, otherwise as a name."""
    spdx = LICENSE_FIELD_TO_SPDX.get(value.strip().lower())
    return {"license": {"id": spdx}} if spdx else {"license": {"name": value.strip()}}


def licenses_for(dist: Distribution) -> list[dict[str, Any]]:
    """Derive CycloneDX license entries from installed distribution metadata."""
    metadata = dist.metadata

    expression = metadata.get("License-Expression")
    if expression:
        return [{"expression": expression.strip()}]

    entries: list[dict[str, Any]] = []
    for classifier in metadata.get_all("Classifier") or []:
        if not classifier.startswith("License :: "):
            continue
        label = classifier.split(" :: ")[-1]
        if label == "OSI Approved":
            continue
        spdx = CLASSIFIER_TO_SPDX.get(label)
        entries.append({"license": {"id": spdx}} if spdx else {"license": {"name": label}})
    if entries:
        return entries

    legacy = (metadata.get("License") or "").strip()
    # Some projects dump the whole license text into this field; keep it out of the SBOM.
    if legacy and "\n" not in legacy and len(legacy) <= 64:
        return [spdx_or_name(legacy)]
    if legacy:
        return [{"license": {"name": "Declared in package metadata (full text)"}}]
    return []


def external_references(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build distribution references, one per locked sdist or wheel, with hashes."""
    references = []
    for artifact in artifacts:
        reference: dict[str, Any] = {"type": "distribution", "url": artifact["url"]}
        digest = artifact.get("hash", "")
        if digest.startswith("sha256:"):
            reference["hashes"] = [
                {"alg": "SHA-256", "content": digest.removeprefix("sha256:")}
            ]
        references.append(reference)
    return references


def enrich_root_component(component: dict[str, Any], project: dict[str, Any]) -> None:
    """Add the metadata uv omits for the project's own root component."""
    component["purl"] = f"pkg:pypi/youtube-to-docs@{project['version']}"
    component["description"] = project["description"]
    # PEP 639: `license` in pyproject.toml is an SPDX license expression.
    component["licenses"] = [{"expression": project["license"]}]
    component["externalReferences"] = [
        {"type": "vcs", "url": REPO_URL},
        {"type": "documentation", "url": DOCS_URL},
        {"type": "distribution", "url": PYPI_URL},
    ]


def main() -> int:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT

    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        project = tomllib.load(f)["project"]

    bom = run_uv_export()
    artifacts = load_lock_artifacts()
    installed = installed_distributions()

    root = bom.get("metadata", {}).get("component")
    if root:
        enrich_root_component(root, project)

    bom.setdefault("metadata", {}).setdefault("tools", []).append(
        {
            "vendor": "DoIT - Artificial Intelligence",
            "name": "scripts/generate_sbom.py",
            "version": project["version"],
        }
    )

    unlicensed: list[str] = []
    for component in bom.get("components", []):
        name = canonical(component["name"])

        dist = installed.get(name)
        if dist:
            entries = licenses_for(dist)
            if entries:
                component["licenses"] = entries
            else:
                unlicensed.append(component["name"])
        else:
            unlicensed.append(f"{component['name']} (not installed)")

        references = external_references(artifacts.get(name, []))
        if references:
            component["externalReferences"] = references

    output_path.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")

    components = bom.get("components", [])
    licensed = sum(1 for c in components if c.get("licenses"))
    print(f"Wrote {output_path.relative_to(REPO_ROOT)}")
    print(f"  CycloneDX {bom['specVersion']}, serial {bom['serialNumber']}")
    print(f"  {len(components)} dependencies, {licensed} with a license declaration")
    if unlicensed:
        print(f"  no license declaration for: {', '.join(sorted(unlicensed))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
