# Optional build CA

No organization-specific certificate is committed here. Set `CURSOR_CA_CERT` to the absolute
path of a PEM root certificate when a TLS-inspecting proxy is present. Docker BuildKit mounts it
ephemerally while downloading Cursor, and Compose mounts it read-only at runtime.
