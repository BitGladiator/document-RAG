FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY start.sh /workspace/start.sh

RUN chmod +x /workspace/start.sh

EXPOSE 8888
EXPOSE 5500

CMD ["/workspace/start.sh"]
