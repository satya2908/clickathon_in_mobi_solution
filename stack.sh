#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

COMPOSE=(docker compose)
ALL_PROFILES=(--profile recommendations --profile selfhosted)

usage() {
  cat <<'EOF'
Usage:
  ./stack.sh                         Interactive menu
  ./stack.sh up [--with-ai]          Start the product, optionally with Cursor recommendations
  ./stack.sh down                    Stop the complete stack
  ./stack.sh status                  Show service status
  ./stack.sh logs [SERVICE]          Follow logs (all services by default)
  ./stack.sh rebuild [--with-ai]     Rebuild product images
  ./stack.sh doctor [--with-ai]      Validate Docker and Compose configuration
  ./stack.sh help

The AI profile requires CURSOR_API_KEY in the environment or root .env. Starting it makes the
UI control available, but the browser toggle remains off until a user explicitly enables it.
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "Docker is not installed or not on PATH."
  docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is unavailable."
  docker info >/dev/null 2>&1 || die "Docker is not running."
}

env_file_value() {
  local key="$1"
  local value=""

  if [[ -f .env ]]; then
    value="$(
      awk -v key="$key" '
        index($0, key "=") == 1 {
          print substr($0, length(key) + 2)
          exit
        }
      ' .env
    )"
  fi
  value="${value%$'\r'}"
  if [[ ${#value} -ge 2 ]]; then
    if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
      value="${value:1:${#value}-2}"
    fi
  fi
  printf '%s' "$value"
}

require_cursor_key() {
  local key="${CURSOR_API_KEY:-}"
  if [[ -z "$key" ]]; then
    key="$(env_file_value CURSOR_API_KEY)"
  fi
  [[ -n "$key" ]] || die \
    "CURSOR_API_KEY is empty. Add it to .env before using --with-ai."
}

validate_cursor_ca() {
  local ca="${CURSOR_CA_CERT:-}"
  if [[ -z "$ca" ]]; then
    ca="$(env_file_value CURSOR_CA_CERT)"
  fi
  if [[ -n "$ca" && ! -r "$ca" ]]; then
    die "CURSOR_CA_CERT is not a readable file: $ca"
  fi
}

compose_for_mode() {
  local with_ai="$1"
  shift

  if [[ "$with_ai" == true ]]; then
    RECOMMENDATIONS_ENABLED=true \
      "${COMPOSE[@]}" --profile recommendations "$@"
  else
    RECOMMENDATIONS_ENABLED=false \
      "${COMPOSE[@]}" "$@"
  fi
}

validate_compose() {
  local with_ai="$1"
  compose_for_mode "$with_ai" config --quiet
}

wait_for_ai() {
  local container_id status
  container_id="$("${COMPOSE[@]}" --profile recommendations ps -q cursor-cli-agent)"
  [[ -n "$container_id" ]] || die "Cursor CLI container was not created."

  printf 'Waiting for Cursor CLI service health'
  for _ in {1..45}; do
    status="$(docker inspect \
      --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
      "$container_id" 2>/dev/null || true)"
    case "$status" in
      healthy)
        printf ' healthy\n'
        return 0
        ;;
      unhealthy|exited|dead)
        printf ' %s\n' "$status"
        "${COMPOSE[@]}" --profile recommendations logs --tail=40 cursor-cli-agent
        return 1
        ;;
      *)
        printf '.'
        sleep 2
        ;;
    esac
  done

  printf ' timed out\n'
  "${COMPOSE[@]}" --profile recommendations logs --tail=40 cursor-cli-agent
  return 1
}

start_stack() {
  local with_ai="$1"
  local web_port="${WEB_PORT:-}"

  require_docker
  if [[ -z "$web_port" ]]; then
    web_port="$(env_file_value WEB_PORT)"
  fi
  web_port="${web_port:-3000}"

  if [[ "$with_ai" == true ]]; then
    require_cursor_key
    validate_cursor_ca
  else
    # Switching from AI mode back to core mode must actually stop the optional service.
    "${COMPOSE[@]}" --profile recommendations stop cursor-cli-agent >/dev/null 2>&1 || true
  fi

  validate_compose "$with_ai"
  compose_for_mode "$with_ai" up -d --build

  if [[ "$with_ai" == true ]]; then
    if ! wait_for_ai; then
      die "Cursor CLI did not become healthy; the core product remains running."
    fi
    printf 'Verdict is running with optional AI recommendations available.\n'
    printf 'Open http://localhost:%s and enable the UI toggle when wanted.\n' \
      "$web_port"
  else
    printf 'Verdict is running at http://localhost:%s (AI recommendations disabled).\n' \
      "$web_port"
  fi
}

rebuild_stack() {
  local with_ai="$1"

  require_docker
  if [[ "$with_ai" == true ]]; then
    require_cursor_key
    validate_cursor_ca
  fi
  validate_compose "$with_ai"
  if [[ "$with_ai" == true ]]; then
    compose_for_mode true build --pull verdict web
    # The official installer resolves a release dynamically, so its cached layer must be skipped
    # when the user explicitly asks for a refresh.
    compose_for_mode true build --pull --no-cache cursor-cli-agent
  else
    compose_for_mode false build --pull
  fi
}

stop_stack() {
  require_docker
  "${COMPOSE[@]}" "${ALL_PROFILES[@]}" down
}

show_status() {
  require_docker
  "${COMPOSE[@]}" "${ALL_PROFILES[@]}" ps
}

follow_logs() {
  require_docker
  if [[ $# -gt 0 ]]; then
    "${COMPOSE[@]}" "${ALL_PROFILES[@]}" logs --tail=200 -f "$1"
  else
    "${COMPOSE[@]}" "${ALL_PROFILES[@]}" logs --tail=200 -f
  fi
}

doctor() {
  local with_ai="$1"
  require_docker
  if [[ "$with_ai" == true ]]; then
    require_cursor_key
    validate_cursor_ca
  fi
  validate_compose "$with_ai"
  printf 'Docker and Compose configuration are valid%s.\n' \
    "$([[ "$with_ai" == true ]] && printf ' for AI mode' || true)"
}

parse_ai_flag() {
  if [[ $# -eq 0 ]]; then
    printf false
  elif [[ $# -eq 1 && "$1" == "--with-ai" ]]; then
    printf true
  else
    die "Expected no option or --with-ai."
  fi
}

interactive_menu() {
  [[ -t 0 ]] || {
    usage
    exit 2
  }

  cat <<'EOF'

Verdict stack
  1) Start product
  2) Start product + Cursor AI recommendations
  3) Stop stack
  4) Show status
  5) Follow logs
  6) Validate configuration
  7) Exit
EOF

  local choice
  read -r -p "Choose [1-7]: " choice
  case "$choice" in
    1) start_stack false ;;
    2) start_stack true ;;
    3) stop_stack ;;
    4) show_status ;;
    5) follow_logs ;;
    6)
      read -r -p "Validate AI mode too? [y/N]: " choice
      [[ "$choice" =~ ^[Yy]$ ]] && doctor true || doctor false
      ;;
    7) exit 0 ;;
    *) die "Unknown menu choice: $choice" ;;
  esac
}

main() {
  local command="${1:-menu}"
  if [[ $# -gt 0 ]]; then
    shift
  fi

  case "$command" in
    menu) interactive_menu ;;
    up) start_stack "$(parse_ai_flag "$@")" ;;
    down)
      [[ $# -eq 0 ]] || die "down does not accept options."
      stop_stack
      ;;
    status)
      [[ $# -eq 0 ]] || die "status does not accept options."
      show_status
      ;;
    logs)
      [[ $# -le 1 ]] || die "logs accepts at most one service name."
      follow_logs "$@"
      ;;
    rebuild) rebuild_stack "$(parse_ai_flag "$@")" ;;
    doctor) doctor "$(parse_ai_flag "$@")" ;;
    help|-h|--help)
      [[ $# -eq 0 ]] || die "help does not accept options."
      usage
      ;;
    *)
      usage >&2
      die "Unknown command: $command"
      ;;
  esac
}

main "$@"
