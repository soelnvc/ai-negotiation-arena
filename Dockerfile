FROM python:3.10-slim

# Keep runtime predictable and logs unbuffered.
ENV PYTHONDONTWRITEBYTECODE=1 \
		PYTHONUNBUFFERED=1 \
		PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy source. Build context should be project root (MetaHack/BoxOne).
COPY . /app

# Install runtime dependencies from the environment package definition.
RUN pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir ./startone

# Run as non-root user for better container security posture.
RUN addgroup --system app && adduser --system --ingroup app app && chown -R app:app /app
USER app

EXPOSE 7860

# Port-level health check without curl dependency.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
	CMD python -c "import os,socket,sys; p=int(os.getenv('PORT','7860')); s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1',p))==0 else 1)"

CMD ["sh", "-c", "uvicorn startone.server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]