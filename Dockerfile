# Debian host; Ubuntu userspace inside container.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=4
ARG COLMAP_REF=4.2.1
ARG OPENMVS_REF=v2.4.0
ARG VCG_REF=658ba36d0a5666650da6e066b4794efc5a463407
ENV QT_QPA_PLATFORM=offscreen
ENV PATH="/usr/local/bin/OpenMVS:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git cmake ninja-build build-essential python3 python3-dev ffmpeg \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libboost-iostreams-dev libboost-serialization-dev libboost-thread-dev \
    libeigen3-dev libopenimageio-dev openimageio-tools libmetis-dev \
    libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
    libceres-dev libcurl4-openssl-dev libssl-dev libopenblas-dev \
    libopencv-dev libpng-dev libjpeg-dev libtiff-dev libglu1-mesa-dev \
    libglfw3-dev libgmp-dev libmpfr-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
# OpenMVS v2.4.0 upstream build recipe uses CGAL 6.0.1.
RUN git clone --depth 1 --branch v6.0.1 https://github.com/CGAL/cgal.git /opt/cgal-src \
    && cmake -S /opt/cgal-src -B /opt/cgal-build -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX=/opt/cgal \
    && cmake --build /opt/cgal-build --parallel ${BUILD_JOBS} \
    && cmake --install /opt/cgal-build
# Explicit sm_75 supports Tesla T4. Image build does not need a GPU.
RUN git clone --depth 1 --branch ${COLMAP_REF} https://github.com/colmap/colmap.git /opt/colmap-src \
    && cmake -S /opt/colmap-src -B /opt/colmap-build -GNinja \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 \
       -DCUDA_ENABLED=ON -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF -DONNX_ENABLED=OFF \
       -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=/opt/cgal \
    && cmake --build /opt/colmap-build --parallel ${BUILD_JOBS} \
    && cmake --install /opt/colmap-build
# OpenMVS v2.4.0 requires nanoflann and JPEG XL (via pkg-config).
# Keep this after COLMAP so the existing COLMAP build layer stays reusable.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnanoflann-dev libjxl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
# OpenCV 4.6 lacks the JPEG XL encoder flag used by OpenMVS v2.4.0.
COPY openmvs-opencv46.patch /tmp/openmvs-opencv46.patch
RUN git init /opt/vcglib \
    && git -C /opt/vcglib remote add origin https://github.com/cdcseacave/VCG.git \
    && git -C /opt/vcglib fetch --depth 1 origin ${VCG_REF} \
    && git -C /opt/vcglib checkout --detach FETCH_HEAD \
    && git clone --depth 1 --branch ${OPENMVS_REF} https://github.com/cdcseacave/openMVS.git /opt/openmvs-src \
    && git -C /opt/openmvs-src apply --check /tmp/openmvs-opencv46.patch \
    && git -C /opt/openmvs-src apply /tmp/openmvs-opencv46.patch \
    && cmake -S /opt/openmvs-src -B /opt/openmvs-build -GNinja \
       -DCMAKE_BUILD_TYPE=Release -DOpenMVS_USE_CUDA=OFF \
       -DOpenMVS_BUILD_VIEWER=OFF -DOpenMVS_USE_PYTHON=OFF -DOpenMVS_USE_BREAKPAD=OFF -DOpenMVS_ENABLE_TESTS=OFF -DOpenMVS_ENABLE_IPO=OFF \
       -DVCG_ROOT=/opt/vcglib -DCMAKE_PREFIX_PATH=/opt/cgal \
    && cmake --build /opt/openmvs-build --parallel ${BUILD_JOBS} \
    && cmake --install /opt/openmvs-build \
    && ldconfig
# Build CUDA-enabled Ceres and relink COLMAP. Earlier successful layers stay cached.
# Ceres 2.2 has only dense CUDA solvers, so global positioning and large bundle adjustments
# fell back to a mostly single-threaded CPU sparse solver. cuDSS (sparse CUDA) first appears in
# Ceres 2.3, which has no release tag yet; pin a master commit.
ARG CERES_GPU_REF=07a952622c2118433c783b1c3bd3fabe68420fc5
RUN apt-get update && apt-get install -y --no-install-recommends \
       libgflags-dev libsuitesparse-dev cudss-cuda-12 \
    && rm -rf /var/lib/apt/lists/* \
    && cudss_dir="$(dirname "$(find /usr -name cudss-config.cmake | head -1)")" \
    && test -f "${cudss_dir}/cudss-config.cmake" \
    && find /usr -name 'libcudss.so*' -printf '%h\n' | sort -u > /etc/ld.so.conf.d/cudss.conf && ldconfig \
    && git init -q /opt/ceres-cuda-src \
    && git -C /opt/ceres-cuda-src fetch -q --depth 1 https://github.com/ceres-solver/ceres-solver.git ${CERES_GPU_REF} \
    && git -C /opt/ceres-cuda-src checkout -q FETCH_HEAD \
    && git -C /opt/ceres-cuda-src submodule update -q --init --depth 1 third_party/abseil-cpp \
    && cmake -S /opt/ceres-cuda-src -B /opt/ceres-cuda-build -GNinja \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/ceres-cuda \
       -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
       -DCMAKE_CUDA_ARCHITECTURES=75 -DWITH_CUDA=default -DWITH_SUITESPARSE=ON -Dcudss_DIR="${cudss_dir}" \
       -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
    && cmake --build /opt/ceres-cuda-build --parallel ${BUILD_JOBS} \
    && cmake --install /opt/ceres-cuda-build \
    && test -f /opt/ceres-cuda/include/ceres/internal/config.h \
    && ! grep -q '^#define CERES_NO_CUDA$' /opt/ceres-cuda/include/ceres/internal/config.h \
    && ! grep -q '^#define CERES_NO_CUDSS$' /opt/ceres-cuda/include/ceres/internal/config.h \
    && ! grep -q '^#define CERES_NO_SUITESPARSE$' /opt/ceres-cuda/include/ceres/internal/config.h
RUN cudss_dir="$(dirname "$(find /usr -name cudss-config.cmake | head -1)")" \
    && cmake -S /opt/colmap-src -B /opt/colmap-build -GNinja -Dcudss_DIR="${cudss_dir}" \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 \
       -DCUDA_ENABLED=ON -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF -DONNX_ENABLED=OFF \
       -DBLA_VENDOR=OpenBLAS -DCeres_DIR=/opt/ceres-cuda/lib/cmake/Ceres \
       -DCMAKE_PREFIX_PATH="/opt/ceres-cuda;/opt/cgal" \
    && cmake --build /opt/colmap-build --parallel ${BUILD_JOBS} \
    && cmake --install /opt/colmap-build \
    && ldconfig
# NVDEC needs the NVIDIA video driver libraries injected into the container.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
# CGAL meshing, texture atlas generation and v360 projection remain CPU work.
COPY mp4_to_model.py /app/mp4_to_model.py
WORKDIR /output
ENTRYPOINT ["python3", "/app/mp4_to_model.py"]
