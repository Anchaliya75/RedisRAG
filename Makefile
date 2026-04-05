.PHONY: setup server ingest search clean

setup:
	bash entrypoint.sh

server: setup
	.venv/bin/uvicorn app:app --reload --port 8000

clean:
	rm -rf .venv
	docker rm -f redis-stack 2>/dev/null || true
	@echo "Cleaned up venv and Redis container."
