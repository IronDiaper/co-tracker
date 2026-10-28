ARG PYTHON_IMAGE=python:3.10.15-slim-bookworm

FROM ${PYTHON_IMAGE} AS weights
ARG CKPT_URL=https://huggingface.co/facebook/cotracker3/resolve/bf55ea50d4390e1820a267f131cd6587240fb2c5/scaled_offline.pth
ARG CKPT_URL_MIRROR=https://www.modelscope.cn/models/facebook/cotracker3/resolve/master/scaled_offline.pth
ARG CKPT_SHA256=2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834
ARG CKPT_SIZE=101890938
COPY scripts/download_weights.py /tmp/download_weights.py
COPY checkpoints/ /tmp/local_ckpt/
RUN python /tmp/download_weights.py \
        --url "${CKPT_URL}" --url "${CKPT_URL_MIRROR}" \
        --sha256 "${CKPT_SHA256}" --size "${CKPT_SIZE}" \
        --local /tmp/local_ckpt/scaled_offline.pth \
        --out /weights/scaled_offline.pth

FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt && pip check

COPY setup.py ./
COPY cotracker ./cotracker
RUN pip install --no-deps -e .

FROM base AS app
COPY --from=weights /weights/scaled_offline.pth /app/checkpoints/scaled_offline.pth
ENV COTRACKER_CHECKPOINT=/app/checkpoints/scaled_offline.pth \
    RESULTS_DIR=/app/results
COPY assets/apple.mp4 ./assets/apple.mp4
COPY scripts ./scripts
RUN mkdir -p /app/results && chmod 777 /app/results

FROM app AS test
COPY requirements-test.txt ./
RUN pip install -r requirements-test.txt && pip check
COPY pytest.ini ./
COPY tests ./tests
CMD ["python", "-m", "pytest"]

FROM app AS runtime
CMD ["python", "scripts/run_demo.py", "--video", "assets/apple.mp4"]
