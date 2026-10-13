FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    NO_PROXY=localhost,127.0.0.1 \
    no_proxy=localhost,127.0.0.1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates curl unzip && \
    rm -rf /var/lib/apt/lists/*

COPY ["AI_单药预测_魔搭Docker部署代码/", "/app/"]

ARG INSTALL_DENSITY_CPU=0

ENV INSTALL_DENSITY_CPU=${INSTALL_DENSITY_CPU} \
    DENSITY_CPU_ADAPTATION=${INSTALL_DENSITY_CPU}

RUN python scripts/fetch_release_assets.py --core && \
    if [ "$INSTALL_DENSITY_CPU" = "1" ]; then \
        python scripts/fetch_release_assets.py --density; \
    fi

RUN python -m pip install --no-cache-dir \
    -r requirements_ui.txt

RUN bash docker/setup_cloud_envs.sh

EXPOSE 7860

CMD ["python", "app.py", "--port", "7860"]
