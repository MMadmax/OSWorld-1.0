#!/usr/bin/env bash
#
# Run OSWorld through a Phoenix sidecar hosted on a trusted machine and
# exposed locally with an SSH reverse tunnel.
#
# Trusted/Toolathlon host:
#   cd /data/lian/Toolathlon
#   bash start_phoenix_model_sidecars.sh start-core
#   ssh -NT -o ExitOnForwardFailure=yes \
#     -R 127.0.0.1:19600:127.0.0.1:19600 root@<osworld-host>
#
# OSWorld host:
#   bash run_osworld_remote_sidecar.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SIDECAR_BASE_URL="${SIDECAR_BASE_URL:-http://127.0.0.1:19600/v1}"
SIDECAR_API_KEY="${SIDECAR_API_KEY:-iai-sidecar}"
MODEL="${MODEL:-glm-5.2}"

HEALTH_URL="${SIDECAR_BASE_URL%/}/health"
COMPLETIONS_URL="${SIDECAR_BASE_URL%/}/chat/completions"

probe_dir="$(mktemp -d)"
cleanup() {
  rm -f "${probe_dir}/health.json" "${probe_dir}/probe.json"
  rmdir "${probe_dir}" 2>/dev/null || true
}
trap cleanup EXIT

if ! curl --silent --show-error --fail --max-time 5 \
  --output "${probe_dir}/health.json" "${HEALTH_URL}"; then
  echo "ERROR: trusted Phoenix sidecar is not reachable at ${HEALTH_URL}." >&2
  echo "Start Toolathlon's sidecar on the trusted host and establish the SSH reverse tunnel shown at the top of this script." >&2
  exit 4
fi

health_ok="$({ python3 - "${probe_dir}/health.json" <<'PY'
import json
import pathlib
import sys

try:
    payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    print("0")
else:
    print("1" if payload.get("ok") is True else "0")
PY
} 2>/dev/null)"
if [[ "${health_ok}" != "1" ]]; then
  echo "ERROR: ${HEALTH_URL} did not return a valid sidecar health response." >&2
  exit 4
fi

if [[ "${SIDECAR_SKIP_PROBE:-0}" != "1" ]]; then
  probe_status="$(curl --silent --show-error --max-time 60 \
    --output "${probe_dir}/probe.json" \
    --write-out '%{http_code}' \
    --request POST "${COMPLETIONS_URL}" \
    --header 'Content-Type: application/json' \
    --header "Authorization: Bearer ${SIDECAR_API_KEY}" \
    --data "{\"model\":\"${MODEL}\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Reply only OK\"}],\"max_tokens\":32,\"temperature\":0}")"

  if ! python3 - "${probe_status}" "${probe_dir}/probe.json" <<'PY'
import json
import pathlib
import sys

status = sys.argv[1]
raw = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
try:
    payload = json.loads(raw)
except Exception:
    print(f"ERROR: sidecar probe returned non-JSON (HTTP {status}).", file=sys.stderr)
    raise SystemExit(1)

choices = payload.get("choices") if isinstance(payload, dict) else None
if status != "200" or not isinstance(choices, list) or not choices:
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("type") or "upstream error")
    else:
        message = "missing choices"
    is_waf = any(marker in message.lower() for marker in ("bxpunish", "bixi-intl", "action=deny", "x5secdata"))
    category = "remote sidecar egress was rejected by WAF" if is_waf else "model probe failed"
    print(f"ERROR: {category} (HTTP {status}): {message[:240]}", file=sys.stderr)
    raise SystemExit(1)

print("Trusted sidecar model probe succeeded.")
PY
  then
    exit 4
  fi
fi

export API_BASE="${SIDECAR_BASE_URL%/}"
export API_KEY="${SIDECAR_API_KEY}"
export MODEL

exec bash "${SCRIPT_DIR}/run_osworld_iai.sh"
