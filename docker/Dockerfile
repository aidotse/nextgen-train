# syntax=docker/dockerfile:1.4
#
# A Dockerfile that encapsulates the entire operative codebase as a standalone environment
#
# docker build --secret id=internal_ca,src=$CERTS --secret id=uv_username,env=$UV_USERNAME --secret id=uv_password,env=$UV_PASSWORD --build-arg VERSION=... -f docker/Dockerfile -t nextgen-train:latest .
#
# CERTS:            <your/PEM/or/CRT/file>. Adding this --secret is optional.
# UV_USERNAME:      Gitlab username
# UV_PASSWORD:      PAT (for this repo)
#
# ---- STAGE 1: Base ------------------------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS base

# Certificates
RUN --mount=type=secret,id=internal_ca \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    if [ -f /run/secrets/internal_ca ]; then \
        cp /run/secrets/internal_ca /usr/local/share/ca-certificates/internal-ca.crt && \
        update-ca-certificates; \
    fi && \
    rm -rf /var/lib/apt/lists/*

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    UV_CERT_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# ---- STAGE 2: Builder ------------------------------------------------------------------------------------------------
FROM base AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Optimizations for uv in build stage
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# Hatchling requires a version number to package / install with the code. Pass via --build-arg
# SETUPTOOLS_SCM_PRETEND_VERSION - tell Hatchling to pretend this is the version instead of looking for .git folder
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION

WORKDIR /app

# Because of --no-install-project, this dependency layer will only invalidate when we install new dependencies, not
# change our source code
# NOTE: adapt dependency groups as required based on e.g. target compute. --group flags MUST match other uv sync calls
COPY pyproject.toml uv.lock ./
RUN --mount=type=secret,id=uv_username \
    --mount=type=secret,id=uv_password \
    --mount=type=cache,target=/root/.cache/uv \
    UV_INDEX_NEXTGEN_USERNAME=$(cat /run/secrets/uv_username) \
    UV_INDEX_NEXTGEN_PASSWORD=$(cat /run/secrets/uv_password) \
    uv sync --frozen --no-dev --group training --no-install-project


COPY README.md LICENSE.md ./
COPY ./scripts ./scripts
COPY ./nextgen_train ./nextgen_train

# Application layer. `--no-editable`: installs our project dependency straight into `site-packages` = no messing around
# with env variables to find the package
# NOTE: adapt dependency groups as required based on e.g. target compute. --group flags MUST match other uv sync calls
RUN --mount=type=secret,id=uv_username \
    --mount=type=secret,id=uv_password \
    --mount=type=cache,target=/root/.cache/uv \
    UV_INDEX_NEXTGEN_USERNAME=$(cat /run/secrets/uv_username) \
    UV_INDEX_NEXTGEN_PASSWORD=$(cat /run/secrets/uv_password) \
    uv sync --frozen --no-dev --group training --no-editable

# ---- STAGE 3: Production ---------------------------------------------------------------------------------------------
FROM base AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    HYDRA_FULL_ERROR=1

WORKDIR /app

COPY --link --from=builder /app/.venv /app/.venv
COPY --link --from=builder /app/scripts /app/scripts
COPY --link --from=builder /app/nextgen_train /app/nextgen_train

ENV CMD=""
CMD exec $CMD
