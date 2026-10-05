# Security and data handling

## Scope

This repository is a portfolio / demonstration project. Public files and Git history were scanned with Gitleaks 8.30.1, and runtime dependency versions were checked against npm / PyPI vulnerability databases during the October 4, 2026 release. These checks find known patterns and advisories; they are not a penetration test or a guarantee that the application is secure.

Keep secrets in private environment variables or a secret manager. `.env.example` is a blank template. Do not commit populated workflow exports, databases, uploads, customer records, transcripts, recordings or provider keys. A public frontend variable is visible to every visitor.

## Reporting

Report a suspected issue privately through the contact channel at [synqlogic.com](https://synqlogic.com). Include the repository, affected component and a minimal synthetic reproduction. Do not put credentials or personal data in a public GitHub issue. Rotate any exposed live secret immediately; removing it from a file does not invalidate it or erase history.

## Automated controls

The security workflow scans Git history on pushes / pull requests with fully redacted output. Workflow permissions are read-only, official checkout is pinned to a commit, and the scanner archive is verified by SHA-256 before execution. Dependency update checks are configured weekly. Review and test updates before merging them.

## Project-specific boundaries

- Requests and retrieved documents can be sent to your Gemini / Groq providers. Use public or approved data and confirm provider handling before adding sensitive documents.
- A shared demo API key is not individual identity or tenant isolation. Authentication uses constant-time comparison; input bounds and one-job locking limit individual requests, but the lock is not a spending / rate cap.
- API schemas are disabled and responses use `no-store`. Provider exception text is omitted from API logs; do not add private documents or credentials to logs.
- LangSmith tracing is opt-in and the Render blueprint defaults it off. Existing Render environment values may still enable tracing; check them separately. Traces can contain inputs, retrieved context and outputs.
- Retrieval, Critic approval and deterministic checks do not guarantee factual accuracy or prevent prompt injection. The independent judge is an evaluation tool, not part of every API request.
