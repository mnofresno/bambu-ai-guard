# Bambu AI Guard hardening plan

## Objective

Make the monitor safe and coherent enough for guarded local operation, while
keeping `auto_pause` disabled by default. This PR must not claim production
readiness without real-printer verification.

## Scope

1. Protect the local HTTP control surface: bind safely by default, add an
   explicit authentication/control boundary, validate all mutable settings,
   prevent XSS, and make enable/shadow/threshold controls actually work.
2. Repair printer and camera lifecycle: MQTT connection state, subscription
   race, publish acknowledgement/error handling, synchronized reports,
   cancelable sockets, stale-frame rejection, and independent frame readers.
3. Make pause transactional: only pause an actively printing job, re-check a
   fresh frame/risk, confirm the printer state, and recover from failures.
4. Validate configuration strictly, including booleans, ranges, required
   secrets, paths, providers, and documented `.env` behavior.
5. Correct ONNX output-shape handling and normalize boxes after letterboxing;
   add tests for both supported output layouts.
6. Make evidence durable and bounded: unique event IDs, after-trigger frames,
   retention/size limits, and configured file logging.
7. Fix dataset split leakage and document the actual capability boundary of
   the default generic model.

## Non-goals

- No cloud deployment or remote printer access.
- No enabling `auto_pause` by default.
- No silent TLS bypass as a security solution; self-signed printer handling
  must be explicit and documented.
- No modification of unrelated user worktree changes.

## Acceptance criteria

- Tests cover API authorization/validation, UI payload behavior, MQTT state,
  stale frames, pause failure/recovery, ONNX layouts/coordinates, recorder
  uniqueness/retention, and config loading.
- Existing tests remain green and the project has a documented validation
  command sequence.
- The PR description distinguishes static/test evidence from live-printer
  verification, which remains a separate manual gate.
