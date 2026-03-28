FROM python:3.10-slim

# Keep runtime predictable and logs unbuffered.
ENV PYTHONDONTWRITEBYTECODE=1 \
		PYTHONUNBUFFERED=1 \
		PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy source. Build context should be project root (MetaHack/BoxOne).
COPY . /app

# Install only runtime dependencies with bounded versions for reproducibility.
RUN pip install --no-cache-dir --upgrade pip && \
		pip install --no-cache-dir \
				"fastapi>=0.110,<1.0" \
				"uvicorn[standard]>=0.30,<1.0" \
				"pydantic>=2.0,<3.0" \
				"python-dotenv>=1.0,<2.0" \
				"google-genai>=0.3,<1.0" \
				"google-generativeai>=0.8,<1.0"

# Run as non-root user for better container security posture.
RUN addgroup --system app && adduser --system --ingroup app app && chown -R app:app /app
USER app

EXPOSE 8000

# Port-level health check without curl dependency.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
	CMD python -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1',8000))==0 else 1)"

CMD ["uvicorn", "startone.server.app:app", "--host", "0.0.0.0", "--port", "8000"]