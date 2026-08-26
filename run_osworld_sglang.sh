#!/usr/bin/env bash
#
# Run OSWorld against an SGLang OpenAI-compatible server through Phoenix:
#
#   OSWorld -> https://phoenix-gw-eval.alibaba.com/eval/v1
#           -> SGLANG_UPSTREAM_ORIGIN
#
# Phoenix's rl-router must already allow SANDBOX_TRAJECTORY_ID to reach the
# supplied SGLang origin.  This mirrors Toolathlon's
# sidecars/openai_to_phoenix_sglang.py headers without requiring a local
# sidecar process.
#
# Example:
#   SANDBOX_TRAJECTORY_ID=<approved-id> \
#   MODEL=qwen3.8-27B \
#   bash run_osworld_sglang.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

SGLANG_UPSTREAM_ORIGIN="${SGLANG_UPSTREAM_ORIGIN:-http://215.66.234.212:30000}"
if [[ "${SGLANG_UPSTREAM_ORIGIN}" != http://* && "${SGLANG_UPSTREAM_ORIGIN}" != https://* ]]; then
  SGLANG_UPSTREAM_ORIGIN="http://${SGLANG_UPSTREAM_ORIGIN}"
fi
SGLANG_UPSTREAM_ORIGIN="${SGLANG_UPSTREAM_ORIGIN%/}"
SGLANG_UPSTREAM_ORIGIN="${SGLANG_UPSTREAM_ORIGIN%/v1}"

if [[ -z "${SANDBOX_TRAJECTORY_ID:-}" ]]; then
  echo "ERROR: SANDBOX_TRAJECTORY_ID is required for Phoenix's SGLang rl-router." >&2
  echo "Use an approved trajectory ID that is allowed to reach ${SGLANG_UPSTREAM_ORIGIN}." >&2
  exit 2
fi

# Toolathlon forwards local /v1/* calls to Phoenix /eval/v1/* and injects
# these routing headers. OSWorld already sends the Phoenix token, domain proxy
# and timeout; mm_agents/agent.py adds X-Backend-TrajectoryID.
PHOENIX_SGLANG_GATEWAY_URL="${PHOENIX_SGLANG_GATEWAY_URL:-https://phoenix-gw-eval.alibaba.com/eval}"
PHOENIX_SGLANG_GATEWAY_URL="${PHOENIX_SGLANG_GATEWAY_URL%/}"
if [[ "${PHOENIX_SGLANG_GATEWAY_URL}" != */v1 ]]; then
  PHOENIX_SGLANG_GATEWAY_URL="${PHOENIX_SGLANG_GATEWAY_URL}/v1"
fi
export API_BASE="${PHOENIX_SGLANG_GATEWAY_URL}"
export API_KEY="${SGLANG_API_KEY:-sglang-sidecar}"
export PHOENIX_DOMAIN_PROXY="${SGLANG_UPSTREAM_ORIGIN}"
export OSWORLD_BACKEND_TRAJECTORY_ID="${SANDBOX_TRAJECTORY_ID}"
export MODEL="${MODEL:-qwen3.8-27B}"

# SGLang's OpenAI API supports SSE. These remain overridable for backends that
# need a larger generation window or a longer Phoenix timeout.
export OSWORLD_LLM_STREAM="${OSWORLD_LLM_STREAM:-1}"
export PHOENIX_EVAL_TIMEOUT="${PHOENIX_EVAL_TIMEOUT:-300}"
export OSWORLD_LLM_REQUEST_TIMEOUT="${OSWORLD_LLM_REQUEST_TIMEOUT:-600}"

echo "Starting OSWorld through Phoenix SGLang routing"
echo "  model:          ${MODEL}"
echo "  Phoenix base:   ${API_BASE}"
echo "  SGLang origin:  ${SGLANG_UPSTREAM_ORIGIN}"
echo "  trajectory id:  configured"

exec bash "${SCRIPT_DIR}/run_osworld_iai.sh"
