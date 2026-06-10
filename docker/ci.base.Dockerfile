# A base Docker image for CI, intended to setup the system environment
#
# docker build --secret id=internal_ca,src=$CERTS -f docker/ci.base.Dockerfile -t "ci-image:latest" .
#
# $CERTS holds your certificate data
FROM python:3.12-slim-bookworm

# system dependencies
RUN --mount=type=secret,id=internal_ca \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        ca-certificates && \
    if [ -f /run/secrets/internal_ca ]; then \
        cp /run/secrets/internal_ca /usr/local/share/ca-certificates/internal-ca.crt && \
        update-ca-certificates; \
    fi && \
    rm -rf /var/lib/apt/lists/*

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    UV_CERT_BUNDLE=/etc/ssl/certs/ca-certificates.crt


# Uv installation
# This will put 'uv' and 'uvx' in /root/.local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# So we can run tasks in CI independently of project installation.
RUN uv tool install poethepoet

# CI optimizations
# Pre-compile uv to save time on first run
# Also tell uv to skip checking for its own updates during CI runs
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_CHECK_UPDATE=false
