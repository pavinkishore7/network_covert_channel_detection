# Base image with build tools needed for liboqs (see docs/PQC_SETUP.md)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake gcc ninja-build libssl-dev git \
    && rm -rf /var/lib/apt/lists/*

# Build liboqs C library (required before liboqs-python will work)
RUN git clone --depth 1 https://github.com/open-quantum-safe/liboqs.git /tmp/liboqs \
    && mkdir /tmp/liboqs/build && cd /tmp/liboqs/build \
    && cmake -GNinja .. && ninja && ninja install \
    && rm -rf /tmp/liboqs
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]
