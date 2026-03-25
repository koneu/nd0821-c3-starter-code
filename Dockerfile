FROM python:3.13-slim

WORKDIR /app

COPY starter/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY starter/ .

RUN pip install --no-cache-dir -e .

RUN python starter/train_model.py

EXPOSE 7860

CMD ["uvicorn", "starter.main:app", "--host", "0.0.0.0", "--port", "7860"]
