# Hugging Face Docker Space for the SuperKart Streamlit app.
#
# Hugging Face no longer accepts `sdk: streamlit` when a Space is created
# programmatically (its own Streamlit template is a Docker Space), so this image
# runs the app on the port Spaces expects (7860).
FROM python:3.12-slim

# Spaces run containers as user id 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app.py superkart_core.py ./
COPY --chown=user data ./data
COPY --chown=user model ./model
COPY --chown=user .streamlit ./.streamlit

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
