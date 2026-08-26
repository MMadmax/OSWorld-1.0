#!/usr/bin/env bash
#
# Run OSWorld through the Phoenix IAI/DashScope OpenAI-compatible gateway.
#
# The Phoenix/IAI defaults below mirror Toolathlon's verified host-side
# DashScope sidecar. They can still be overridden through environment
# variables when needed.
#
# Example smoke run (one domain, one Docker VM):
#   DOMAIN=chrome NUM_ENVS=1 bash run_osworld_iai.sh
#
# LLM calls happen on the OSWorld host, not inside the desktop VM.

set -Eeuo pipefail

export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"

# Optionally load locally managed credentials. Explicit environment values
# supplied to this launcher take precedence over values sourced from the file.
SECRETS_FILE="${SECRETS_FILE:-${HOME}/.wcb_secretsc}"
API_KEY_OVERRIDE="${API_KEY:-}"
EMP_ID_OVERRIDE="${EMP_ID:-}"
IAI_TAG_OVERRIDE="${IAI_TAG:-}"
API_BASE_OVERRIDE="${API_BASE:-}"
if [[ -f "${SECRETS_FILE}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${SECRETS_FILE}"
  set -u
fi
[[ -n "${API_KEY_OVERRIDE}" ]] && API_KEY="${API_KEY_OVERRIDE}"
[[ -n "${EMP_ID_OVERRIDE}" ]] && EMP_ID="${EMP_ID_OVERRIDE}"
[[ -n "${IAI_TAG_OVERRIDE}" ]] && IAI_TAG="${IAI_TAG_OVERRIDE}"
[[ -n "${API_BASE_OVERRIDE}" ]] && API_BASE="${API_BASE_OVERRIDE}"

# ─── Model gateway ──────────────────────────────────────────────────────────
# Do not enable shell xtrace around this block: it contains credentials.
export PHOENIX_TARGET="${PHOENIX_TARGET:-${API_BASE_OVERRIDE:-http://phoenix-gw-eval.alibaba.com/eval/dashscope}}"
export PHOENIX_TENANT="${PHOENIX_TENANT:-icbu-dashscope-buyer-agent-algo}"
export PHOENIX_EVAL_TOKEN="${PHOENIX_EVAL_TOKEN:-feccf2b0b1fb87a87faad9ac201744bd}"
export PHOENIX_DOMAIN_PROXY="${PHOENIX_DOMAIN_PROXY:-http://iai.vipserver:7001}"
export PHOENIX_EVAL_TIMEOUT="${PHOENIX_EVAL_TIMEOUT:-1200}"
# Bound the host-side HTTP call as well; the gateway timeout header alone does
# not protect workers from a connection that remains open without a response.
export OSWORLD_LLM_REQUEST_TIMEOUT="${OSWORLD_LLM_REQUEST_TIMEOUT:-1230}"
# iai.alibaba-inc.com's unified ingress cuts off non-streaming requests after
# 90 seconds. SSE keeps the connection active as model chunks arrive.
export OSWORLD_LLM_STREAM="${OSWORLD_LLM_STREAM:-1}"
# Model transport failures surfaced as empty responses retry the same benchmark step. Exhausted
# retries mark the task as an error instead of producing a misleading 0 score.
export OSWORLD_LLM_MAX_RETRIES_PER_STEP="${OSWORLD_LLM_MAX_RETRIES_PER_STEP:-3}"
export OSWORLD_LLM_RETRY_BACKOFF_SECONDS="${OSWORLD_LLM_RETRY_BACKOFF_SECONDS:-2}"

MODEL="${MODEL:-qwen3.8-max}"
API_BASE="${API_BASE_OVERRIDE:-${PHOENIX_TARGET}}"
# An explicitly supplied API_KEY wins. Values incidentally sourced from the
# optional secrets file do not replace Phoenix's required pseudo-tenant.
API_KEY="${API_KEY_OVERRIDE:-${PHOENIX_TENANT}}"
EMP_ID="${EMP_ID:-${PHOENIX_EMP_ID:-547066}}"
IAI_TAG="${IAI_TAG:-${PHOENIX_IAI_TAG:-ale-test}}"

if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: API_KEY is required and must be the correct tenant for ${MODEL}." >&2
  exit 2
fi
if [[ -z "${EMP_ID}" ]]; then
  echo "ERROR: EMP_ID is required; use your own employee id." >&2
  exit 2
fi
if [[ -z "${IAI_TAG}" ]]; then
  echo "ERROR: IAI_TAG is required and must identify this evaluation clearly." >&2
  exit 2
fi
if [[ "${API_BASE}" == *"phoenix-gw-eval.alibaba.com"* && -z "${PHOENIX_EVAL_TOKEN}" ]]; then
  echo "ERROR: PHOENIX_EVAL_TOKEN is required for the Phoenix gateway." >&2
  exit 2
fi

# PromptAgent reads these without putting credentials on the process command
# line. The exact endpoint avoids incorrectly inserting /v1 for IAI.
export OPENAI_API_KEY="${API_KEY}"
export OPENAI_BASE_URL="${API_BASE%/}"
export OSWORLD_OPENAI_CHAT_COMPLETIONS_URL="${CHAT_COMPLETIONS_URL:-${API_BASE%/}/chat/completions}"
export OSWORLD_OPENAI_COMPATIBLE=1
export OSWORLD_EMP_ID="${EMP_ID}"
export OSWORLD_IAI_TAG="${IAI_TAG}"

# ─── OSWorld runtime ────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-/data/miniforge3/envs/osworld/bin/python}"
PROVIDER_NAME="${PROVIDER_NAME:-docker}"
PATH_TO_VM="${PATH_TO_VM:-/data/osworld/docker_vm_data/Ubuntu.qcow2}"
CLIENT_PASSWORD="${CLIENT_PASSWORD:-password}"

OBSERVATION_TYPE="${OBSERVATION_TYPE:-screenshot}"
ACTION_SPACE="${ACTION_SPACE:-pyautogui}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-131072}"
MAX_STEPS="${MAX_STEPS:-200}"
HISTORY_N="${HISTORY_N:-100}"
IMAGE_MAX="${IMAGE_MAX:-20}"
FOLD_SIZE="${FOLD_SIZE:-10}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
ENVIRONMENT_RETRIES="${ENVIRONMENT_RETRIES:-2}"
SLEEP_AFTER_EXECUTION="${SLEEP_AFTER_EXECUTION:-0}"

TEST_META_PATH="${TEST_META_PATH:-evaluation_examples/test_nogdrive.json}"
DOMAIN="${DOMAIN:-all}"
# run_multienv.py starts one worker process and one desktop VM per environment.
NUM_ENVS="${NUM_ENVS:-2}"
RESULT_DIR="${RESULT_DIR:-/data/fengruixiang/CUA/OSWorld/output}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

SCREEN_WIDTH="${SCREEN_WIDTH:-1920}"
SCREEN_HEIGHT="${SCREEN_HEIGHT:-1080}"

# ─── Preconditions ──────────────────────────────────────────────────────────
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python environment not found: ${PYTHON_BIN}" >&2
  exit 3
fi
if [[ "${PROVIDER_NAME}" == "docker" ]]; then
  if [[ ! -r /dev/kvm || ! -w /dev/kvm ]]; then
    echo "ERROR: /dev/kvm must be readable and writable for Docker evaluation." >&2
    exit 3
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is unavailable." >&2
    exit 3
  fi
  if [[ ! -f "${PATH_TO_VM}" ]]; then
    echo "ERROR: OSWorld VM image not found: ${PATH_TO_VM}" >&2
    exit 3
  fi
fi
if [[ ! -f "${REPO_ROOT}/${TEST_META_PATH}" && ! -f "${TEST_META_PATH}" ]]; then
  echo "ERROR: task metadata file not found: ${TEST_META_PATH}" >&2
  exit 3
fi

mkdir -p "${RESULT_DIR}" "${REPO_ROOT}/logs"

RUN_ARGS=(
  --provider_name "${PROVIDER_NAME}"
  --path_to_vm "${PATH_TO_VM}"
  --headless
  --action_space "${ACTION_SPACE}"
  --observation_type "${OBSERVATION_TYPE}"
  --model "${MODEL}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --max_tokens "${MAX_TOKENS}"
  --max_steps "${MAX_STEPS}"
  --history_n "${HISTORY_N}"
  --image_max "${IMAGE_MAX}"
  --fold_size "${FOLD_SIZE}"
  --enable_thinking
  --reasoning_effort "${REASONING_EFFORT}"
  --environment_retries "${ENVIRONMENT_RETRIES}"
  --coord relative
  --base_url "${API_BASE%/}"
  --api_key "${API_KEY}"
  --no-enable_proxy
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION}"
  --test_all_meta_path "${TEST_META_PATH}"
  --domain "${DOMAIN}"
  --num_envs "${NUM_ENVS}"
  --result_dir "${RESULT_DIR}"
  --client_password "${CLIENT_PASSWORD}"
  --screen_width "${SCREEN_WIDTH}"
  --screen_height "${SCREEN_HEIGHT}"
  --log_level "${LOG_LEVEL}"
)

cd "${REPO_ROOT}"

echo "Starting OSWorld evaluation"
echo "  model:       ${MODEL}"
echo "  api base:    ${API_BASE}"
echo "  streaming:   ${OSWORLD_LLM_STREAM}"
echo "  provider:    ${PROVIDER_NAME}"
echo "  tasks:       ${TEST_META_PATH} (domain=${DOMAIN})"
echo "  concurrency: ${NUM_ENVS}"
echo "  results:     ${RESULT_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Command:'
  printf ' %q' "${PYTHON_BIN}" scripts/python/run_multienv_qwen.py "${RUN_ARGS[@]}"
  printf '\n'
  exit 0
fi

exec "${PYTHON_BIN}" scripts/python/run_multienv_qwen.py "${RUN_ARGS[@]}"
