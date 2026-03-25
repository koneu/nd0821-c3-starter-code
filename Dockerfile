FROM python:3.13-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

RUN pip install --no-cache-dir --user -r starter/requirements.txt

RUN python starter/starter/train_model.py

EXPOSE 7860

CMD ["uvicorn", "starter.starter.main:app", "--host", "0.0.0.0", "--port", "7860"]
