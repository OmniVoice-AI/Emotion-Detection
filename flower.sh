#!/bin/bash
celery -A app.celery_worker flower --port=5555 --broker=redis://localhost:6379/0