# Cloud deployment: separate Python environments, no retraining.
# IMPORTANT: source build has not been fully tested on ModelScope's builder.
FROM continuumio/miniconda3:latest
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    NO_PROXY=localhost,127.0.0.1 \
    no_proxy=localhost,127.0.0.1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl unzip && rm -rf /var/lib/apt/lists/*
COPY . /app
# Fetch signed release assets, not training data from a different source.
ARG INSTALL_DENSITY_CPU=0
ENV INSTALL_DENSITY_CPU=${INSTALL_DENSITY_CPU} \
    DENSITY_CPU_ADAPTATION=${INSTALL_DENSITY_CPU}
RUN python /app/scripts/fetch_release_assets.py --core && if [ "$INSTALL_DENSITY_CPU" = "1" ]; then python /app/scripts/fetch_release_assets.py --density; fi
# Install web UI in the base Python; ML workers live in isolated Conda prefixes.
RUN python -m pip install --no-cache-dir -r requirements_ui.txt
RUN bash /app/docker/setup_cloud_envs.sh
# Distinct environments are detected by scripts/match_and_run_v06.py at runtime.
EXPOSE 7860
CMD ["python", "app.py", "--port", "7860"]
