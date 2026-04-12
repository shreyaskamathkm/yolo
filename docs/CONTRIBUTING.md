# Contributing to YOLO

Thank you for your interest in contributing to this project! We value your contributions and want to make the process as easy and enjoyable as possible. Below you will find the guidelines for contributing.

## Quick Links
- [Main README](../README.md)
- [License](../LICENSE)
- [Issue Tracker](https://github.com/shreyaskamathkm/yolo/issues)
- [Pull Requests](https://github.com/shreyaskamathkm/yolo/pulls)

## Testing and Formatting
We strive to maintain a high standard of quality in our codebase:
- **Testing:** We use `pytest` for testing. Please add tests for new code you create.
- **Formatting:** Our code follows a consistent style enforced by `isort` for imports sorting and `black` for code formatting. Run these tools to format your code before submitting a pull request.

## CI Workflows

All workflows live in `.github/workflows/`. Shared logic is defined once in reusable workflows (prefixed `_`) and called from the top-level workflows.

| File | Trigger | What it does |
|---|---|---|
| `pr.yaml` | PR to `main` | Version check → lint/test + validation/inference |
| `ci.yaml` | Push to `main` | Lint + test after merge |
| `integration.yaml` | Push to `main` | Validation + inference after merge |
| `release.yaml` | Push to `main` | Creates git tag and GitHub Release |
| `docker.yaml` | GitHub Release published | Builds and pushes Docker image to Docker Hub |
| `docs.yaml` | Push to `main` (docs changed) | Deploys docs to GitHub Pages |

Reusable (internal, not triggered directly):

| File | Used by |
|---|---|
| `_lint-test.yaml` | `pr.yaml`, `ci.yaml` |
| `_validate-inference.yaml` | `pr.yaml`, `integration.yaml` |

### PR check order

```
PR opened
├── version-check              (fast gate, ~5s)
│     ✅ pass
│     ├── lint_and_test        (lint + test)
│     └── validate_and_infer   (validation + inference, Python 3.8 + 3.10)
│           all ✅ → merge allowed
│           any ❌ → merge blocked
```

Tests are skipped entirely if the version wasn't bumped — no CI minutes are wasted.

### After merge

```
PR merged to main
├── ci.yaml          → lint + test
├── integration.yaml → validation + inference
├── release.yaml     → git tag + GitHub Release
└── docker.yaml      → Docker Publish (triggered by the GitHub Release)
```

## How to Contribute

### Proposing Enhancements
For feature requests or improvements, open an issue with:
- A clear title and description.
- Explain why this enhancement would be useful.
- Considerations or potential implementation details.

## Versioning and Releases

This project uses [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).

**Every PR to `main` must include a version bump** in `pyproject.toml`. A CI check will block merge if the version is unchanged.

Choose the bump type based on what your PR does:

| Change type | Command | Example |
|---|---|---|
| Bug fix or small improvement | `make bump-patch` | `0.1.0 → 0.1.1` |
| New feature (backwards-compatible) | `make bump-minor` | `0.1.0 → 0.2.0` |
| Breaking change | `make bump-major` | `0.1.0 → 1.0.0` |

You can also edit `version` in `pyproject.toml` directly if you prefer. Once the PR is merged, a GitHub Release and git tag are created automatically.

## Pull Request Checklist
Before sending your pull request, always check the following:
- The code follows the [Python style guide](https://www.python.org/dev/peps/pep-0008/).
- Code and files are well organized.
- All tests pass.
- New code is covered by tests.
- `version` in `pyproject.toml` has been bumped.
- We would be very happy if [gitmoji😆](https://www.npmjs.com/package/gitmoji-cli) could be used to assist the commit message💬!

## Code Review Process
Once you submit a PR, maintainers will review your work, suggest changes if necessary, and merge it once it’s approved. On merge, a release is created automatically from the version in `pyproject.toml`.

---

Your contributions are greatly appreciated and vital to the project's success!

Please feel free to open an [issue](https://github.com/shreyaskamathkm/yolo/issues) or start a [discussion](https://github.com/shreyaskamathkm/yolo/discussions)!
