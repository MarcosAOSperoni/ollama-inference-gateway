FROM python:3.11-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless wget && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

RUN pip install --no-cache-dir apache-flink==1.18.1 requests

RUN mkdir -p /opt/flink-jars && \
    wget -q "https://repo1.maven.apache.org/maven2/org/apache/flink/flink-connector-kafka/3.1.0-1.18/flink-connector-kafka-3.1.0-1.18.jar" \
        -O /opt/flink-jars/flink-connector-kafka.jar && \
    wget -q "https://repo1.maven.apache.org/maven2/org/apache/kafka/kafka-clients/3.4.0/kafka-clients-3.4.0.jar" \
        -O /opt/flink-jars/kafka-clients.jar

COPY inference_stats.py /app/
WORKDIR /app

CMD ["python", "inference_stats.py"]
