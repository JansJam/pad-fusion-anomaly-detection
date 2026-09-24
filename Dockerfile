FROM python:3.13.15-slim-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt requirements-torch.txt constraints.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt -c constraints.txt && \
    python -m pip install --no-cache-dir --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-torch.txt -c constraints.txt && \
    python -m pip check
COPY infer.py ./
COPY pad ./pad
COPY artifacts ./artifacts
RUN test -s artifacts/fusion_model.pt && python -c "from pad.model import PADPredictor; PADPredictor('artifacts/fusion_model.pt')"
RUN groupadd --gid 10001 paduser && \
    useradd --uid 10001 --gid 10001 --create-home paduser

ENV HOME=/home/paduser USER=paduser LOGNAME=paduser

USER paduser

ENTRYPOINT ["python", "infer.py"]
CMD ["--help"]