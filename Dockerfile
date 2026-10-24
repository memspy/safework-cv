FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 safework \
    && mkdir -p /home/safework/.cache/safework_cv /data \
    && chown -R safework:safework /home/safework /data

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \
    && rm -rf /build

USER safework
WORKDIR /data
ENV SAFEWORK_CV_HOME=/home/safework/.cache/safework_cv

ENTRYPOINT ["safework-cv"]
CMD ["--help"]
