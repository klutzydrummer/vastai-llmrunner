FROM python:3.13-alpine
RUN adduser -D -u 10001 app
WORKDIR /app
COPY server.py index.html app.js endpoints.json ./
USER app
ENV VIDEOTEST_PORT=18810 VIDEOTEST_BASE_HOME=http://videotest-llamaswap:8080/v1
EXPOSE 18810
CMD ["python3", "-u", "server.py"]
