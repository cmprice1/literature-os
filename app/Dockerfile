FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 streamlit
RUN mkdir -p /home/streamlit/.streamlit
RUN chown -R streamlit:streamlit /home/streamlit

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R streamlit:streamlit /app

USER streamlit

ENV STREAMLIT_HOME_PATH=/home/streamlit/.streamlit

CMD ["streamlit", "run", "streamlit_app.py"]
