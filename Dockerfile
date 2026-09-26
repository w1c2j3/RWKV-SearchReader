FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY static ./static
RUN pip install --no-cache-dir . && useradd --uid 10001 --create-home reader && mkdir /data && chown reader:reader /data
USER reader
ENV SEARCH_READER_TRACE_DIRECTORY=/data SEARCH_READER_API_HOST=0.0.0.0
EXPOSE 8790
CMD ["rwkv-search-reader-api"]
