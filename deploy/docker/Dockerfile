# GeoParcel AI Streamlit demo as a Docker image (any Docker host: Render, Railway, a VM, or a paid HF Docker Space).
# Builds the Streamlit app from the GitHub repo at a pinned commit and downloads the model weights
# from the GitHub release, checking their MD5. To deploy a newer version, change REF and rebuild.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

ARG REPO=https://github.com/OMMEOW/SIH26_PS26012
ARG REF=main
# sparse checkout: only what the app needs (skips papers/, video/, results/)
RUN git clone --depth 1 --filter=blob:none --no-checkout "$REPO" . \
    && git sparse-checkout set --no-cone /demo_app.py /src/ /demo_inputs/ /.streamlit/ /requirements-app.txt \
    && git fetch --depth 1 origin "$REF" && git checkout -q FETCH_HEAD && rm -rf .git

RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-app.txt

ARG WEIGHTS=https://github.com/OMMEOW/SIH26_PS26012/releases/download/weights-v1
RUN mkdir -p checkpoints \
    && curl -fsSL -o checkpoints/best_model_ms.pt "$WEIGHTS/best_model_ms.pt" \
    && curl -fsSL -o checkpoints/best_model.pt "$WEIGHTS/best_model.pt" \
    && printf '3e66c7ade418a33cbf02f403c79837ae  checkpoints/best_model_ms.pt\n870e697f60a50139edb226444df9d590  checkpoints/best_model.pt\n' | md5sum -c -

RUN chown -R user:user /home/user
USER user
ENV HOME=/home/user GEOPARCEL_PUBLIC=1 PYTHONUNBUFFERED=1
EXPOSE 7860
# XSRF/CORS are off because Spaces serves the app inside an iframe on another origin,
# where Streamlit's XSRF check rejects file uploads.
CMD ["streamlit", "run", "demo_app.py", "--server.port", "7860", "--server.address", "0.0.0.0", \
     "--server.maxUploadSize", "25", "--server.enableXsrfProtection", "false", "--server.enableCORS", "false", \
     "--browser.gatherUsageStats", "false"]
