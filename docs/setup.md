# NextGen-Train

A repo with some use cases for training models using the [NextGenFramework](https://gitlab.mgmt.ai.se/nextgeninfra/nextgen_framework) decentralized learning framework. The idea behind this project is to highlight how the NextGenFramework decentralized learning framework works and can be used in action for training models on existing datasets.

This repo was created using the [PyMaxQ template](https://gitlab.mgmt.ai.se/dev-tools/pymaxq), please refer to the documentation there for in-depth explanations on the tooling system.

## Setup

This project uses `uv` for Python dependency management and has NextGenFramework as a dependency, which by default communicates with a GitLab instance for central state management. We note that users of this repo are not necessarily vendor-locked to Gitlab; any communication with the backend is abstracted away by NextGenFramework.

### 1. Install `uv`

If you don't have `uv` installed locally, install it via curl (macOS/Linux):

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

*(For Windows or other methods, see the [official uv docs](https://docs.astral.sh/uv/getting-started/installation/)).*

### 2. Authentication

#### NextGenFramework & Gitlab

You will need a GitLab Personal Access Token (PAT) with at least `read_api` access. You can e.g. put these credentials in a local `.env` file in the root of your project (this file is git-ignored), along with the credentials required for authentication with Gitlab for pushing and pulling models:

Create a `.env` file in the root of the project (this file is git-ignored) and add your credentials:

```env
UV_INDEX_NEXTGEN_USERNAME=your_gitlab_username
UV_INDEX_NEXTGEN_PASSWORD=your_personal_access_token
MLFLOW_TRACKING_URL=...
MLFLOW_TRACKING_PASSWORD=...
```

`MLFLOW_TRACKING_URL` is the URL to the Gitlab repo to use, i.e. where to pull and push models from. Example: `https://gitlab.mgmt.ai.se/api/v4/projects/71/ml/mlflow/`. You can find your project ID (in the example: 71) in the main page of your repo. You can find the models from the run below under `Deploy -> Model Registry`.

`MLFLOW_TRACKING_PASSWORD` is an access token with `api` scope and with at least a `Developer` role for that repo.

Note that the `uv` config in your `pyproject.toml` should in this case look like this (the `name` must match the source substring in the env variables):

```bash
[[tool.uv.index]]
name = "nextgen"
url = ...
```

#### CI/CD

To authorize this project to pull the NextGenFramework repo in CI/CD, an additional step is needed: ask the NextGenFramework admins to add this project to the "whitelist" of projects that can pull it, or ask them for a Deploy Token that you can add as a CI/CD variable.

#### HuggingFace

You will need to authenticate with HuggingFace if you are using (the default) datasets stored there:

```bash
huggingface-cli login
```

### 3. Sync the Environment

Source your environment variables and let `uv` build your local `.venv`. To ensure you have all the tools needed for local development (including testing and linting), you can sync with the `dev` group explicitly. Also have a look at what other dependency groups are available, and install the ones that are relevant for your compute if e.g. training locally:
```bash
source .env
uv sync --group dev --group training --group gaudi
```
*(Note: `uv sync` automatically creates the `.venv` folder and installs the project in editable mode).*

### 4. Activate the Virtual Environment

Activate the environment to start using the installed tools (or alternatively, use `uv run...` before all CLI calls)

```bash
source .venv/bin/activate
```

### 5. Install the Task Runner (optional but recommended)

We use `poethepoet` for running project tasks (like bumping versions or running tests). You can install it globally via `uv` so it's always available in your CLI, independently of whether the rest of dependencies are installed:

```bash
uv tool install poethepoet
```

You can now run tasks simply by typing `poe <task_name>` (e.g., `poe test`). The list of all runnable tasks are contained in the `pyproject.toml` file.

### 6. Install pre-commit hooks

Make sure pre-commit hooks are installed before committing anything.

```bash
uv run pre-commit install
```

### 7. Unit testing

To run all tests in `test/`:

```bash
uv run poe test
```

### 8. Logging
By default we use [MLFlow's tracking API](https://mlflow.org/docs/latest/tracking.html) to log experiment parameters and outputs to a local store.

To view experiment runs and results, first make sure that the MLFlow server is running on the machine above if its not already, and point it to wherever your local store (`./mlruns` is located):

```bash
uv run mlflow server --backend-store-uri ./mlruns/ --port 5001
```

Then, use port-forwarding to make the chosen port visible on your localhost. You can do this e.g. with VS Code directly (just manually `Add Port` next to the debug console), or with SSH:

```bash
ssh <VM hostname/IP> -L 5001:127.0.0.1:5001
```

Then just open at browser at [localhost:5000](localhost:5000)

For running experiments, you can either pass the full path to the local store (the `mlruns` directory), or you can also use the URL directly in the command line configuration options, see e.g. `Pharmaceutical Question-Answering Model Training`:

```bash
logger.mlruns="http://localhost:5001"
```
