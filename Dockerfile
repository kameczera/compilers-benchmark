# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG TVM_COMMIT=3b60f1c9b8907dcf5d39a033876020e96e6915b2

FROM ${CUDA_IMAGE} AS tvm-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG TVM_COMMIT
ARG TVM_BUILD_JOBS=6

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        llvm-18-dev \
        ninja-build \
        pkg-config \
        zlib1g-dev \
        libxml2-dev \
        libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/apache/tvm.git /opt/tvm \
    && git -C /opt/tvm checkout "${TVM_COMMIT}" \
    && git -C /opt/tvm submodule update --init --recursive --depth 1

RUN cmake -S /opt/tvm -B /opt/tvm/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_CUDA=/usr/local/cuda \
        -DUSE_CUDNN=ON \
        -DUSE_LLVM="/usr/bin/llvm-config-18 --link-shared" \
        -DUSE_LIBBACKTRACE=AUTO \
    && cmake --build /opt/tvm/build \
        --target tvm tvm_runtime \
        --parallel "${TVM_BUILD_JOBS}"

# O pacote Python do TVM 0.22 carrega a extensao Cython `tvm.ffi.core`, que o
# build CMake nao produz: sem compila-la in-place, `import tvm` falha com
# "cannot import name 'core' from partially initialized module 'tvm.ffi'".
# A extensao acompanha /opt/tvm/python no estagio de runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/ffi-build \
    && /opt/ffi-build/bin/pip install --no-cache-dir "setuptools>=61" "cython==3.1.2" \
    && cd /opt/tvm/python \
    && TVM_FFI=cython /opt/ffi-build/bin/python setup.py build_ext --inplace \
    && ls /opt/tvm/python/tvm/ffi/core*.so


FROM ${CUDA_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.title="CNN Compilers Benchmark" \
      org.opencontainers.image.description="Reproducible TorchInductor, XLA and TVM GPU benchmark" \
      org.opencontainers.image.source="cnnbench"

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
    TVM_HOME=/opt/tvm \
    PYTHONPATH=/opt/tvm/python:/opt/tvm/build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        libllvm18 \
        libxml2 \
        libzstd1 \
        make \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=tvm-builder /opt/tvm/python /opt/tvm/python
COPY --from=tvm-builder /opt/tvm/build/libtvm.so /opt/tvm/build/libtvm.so
COPY --from=tvm-builder /opt/tvm/build/libtvm_runtime.so /opt/tvm/build/libtvm_runtime.so

WORKDIR /opt/cnnbench

COPY envs/requirements_xla.txt /tmp/requirements_xla.txt
COPY envs/requirements_tvm.txt /tmp/requirements_tvm.txt

RUN python3 -m venv /opt/venvs/xla \
    && /opt/venvs/xla/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/venvs/xla/bin/python -m pip install -r /tmp/requirements_xla.txt \
    && /opt/venvs/xla/bin/python -m pip check

RUN python3 -m venv /opt/venvs/tvm \
    && /opt/venvs/tvm/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/venvs/tvm/bin/python -m pip install -r /tmp/requirements_tvm.txt \
    && /opt/venvs/tvm/bin/python -m pip check \
    && rm -f /tmp/requirements_xla.txt /tmp/requirements_tvm.txt

COPY __init__.py collect.py run_bench.py Makefile README.md LICENSE CITATION.cff ARTIFACT_EVALUATION.md ./
COPY backends ./backends
COPY envs ./envs
COPY models ./models
COPY scripts ./scripts
COPY docker/entrypoint.sh ./docker/entrypoint.sh
COPY docs/sblp/main.tex ./docs/sblp/main.tex
COPY docker/entrypoint.sh /usr/local/bin/cnnbench

RUN chmod +x /usr/local/bin/cnnbench scripts/*.sh \
    && /opt/venvs/xla/bin/python -m compileall -q . \
    && /opt/venvs/xla/bin/python -c "import torch, jax; print('torch', torch.__version__); print('jax', jax.__version__)" \
    && test -s /opt/tvm/build/libtvm.so \
    && test -s /opt/tvm/build/libtvm_runtime.so \
    && ls /opt/tvm/python/tvm/ffi/core*.so \
    && mkdir -p /tmp/cuda-stub \
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 \
    && LD_LIBRARY_PATH=/tmp/cuda-stub \
       /opt/venvs/tvm/bin/python -c "import tvm; from tvm import relax; print('tvm', tvm.__version__)" \
    && rm -rf /tmp/cuda-stub \
    && /opt/venvs/xla/bin/python scripts/check_fold.py

VOLUME ["/artifacts"]
ENTRYPOINT ["cnnbench"]
CMD ["help"]
