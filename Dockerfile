# WindGap web app (server.py) in a container - works on any Docker host.
#   docker build -t windgap . && docker run -p 7860:7860 windgap   ->  http://localhost:7860
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 PORT=7860 MPLCONFIGDIR=/tmp/matplotlib

# run as an unprivileged user
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user server.py ./
COPY --chown=user src ./src
COPY --chown=user templates ./templates
COPY --chown=user data ./data

USER user
EXPOSE 7860
CMD ["python", "server.py"]
