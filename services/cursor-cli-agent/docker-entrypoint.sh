#!/bin/sh

set -eu

extra_ca=/run/secrets/cursor_extra_ca
if [ -s "$extra_ca" ] && grep -q "BEGIN CERTIFICATE" "$extra_ca"; then
  bundle=/data/runtime/cursor-ca-bundle.crt
  mkdir -p "$(dirname "$bundle")"
  cat /etc/ssl/certs/ca-certificates.crt "$extra_ca" >"$bundle"
  export SSL_CERT_FILE="$bundle"
  export CURL_CA_BUNDLE="$bundle"
  export NODE_EXTRA_CA_CERTS="$extra_ca"
fi

exec "$@"
