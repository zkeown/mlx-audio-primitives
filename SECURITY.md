# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in mlx-audio-primitives, please report it responsibly.

### How to Report

**Preferred:** Use [GitHub Security Advisories](https://github.com/zkeown/mlx-audio-primitives/security/advisories/new) to report vulnerabilities privately.

This allows us to:
- Discuss the issue privately
- Develop a fix before public disclosure
- Credit you for the discovery (if desired)

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (optional)

### Response Timeline

- **Initial response:** Within 48 hours
- **Status update:** Within 7 days
- **Fix timeline:** Depends on severity and complexity

### Scope

This security policy covers:
- The `mlx-audio-primitives` Python package
- C++ extensions and Metal kernels
- Build and distribution infrastructure

### Out of Scope

- Vulnerabilities in dependencies (report to the respective projects)
- Issues in MLX itself (report to [ml-explore/mlx](https://github.com/ml-explore/mlx))

## Security Best Practices

When using mlx-audio-primitives:
- Keep dependencies updated
- Use virtual environments
- Validate input audio files from untrusted sources
