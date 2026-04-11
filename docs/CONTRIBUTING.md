# Contributing to YOLO

Thank you for your interest in contributing to this project! We value your contributions and want to make the process as easy and enjoyable as possible. Below you will find the guidelines for contributing.

## Quick Links
- [Main README](../README.md)
- [License](../LICENSE)
- [Issue Tracker](https://github.com/WongKinYiu/yolov9mit/issues)
- [Pull Requests](https://github.com/WongKinYiu/yolov9mit/pulls)

## Testing and Formatting
We strive to maintain a high standard of quality in our codebase:
- **Testing:** We use `pytest` for testing. Please add tests for new code you create.
- **Formatting:** Our code follows a consistent style enforced by `isort` for imports sorting and `black` for code formatting. Run these tools to format your code before submitting a pull request.

## GitHub Actions
We utilize GitHub Actions for continuous integration. When you submit a pull request, automated tests, formatting checks, and a version bump check will run. Ensure that all checks pass for your pull request to be accepted.

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

Please feel free to contact [henrytsui000@gmail.com](mailto:henrytsui000@gmail.com)!
