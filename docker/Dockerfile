# syntax = docker/dockerfile:1.2

FROM archlinux

# install packages
#RUN --mount=type=cache,sharing=locked,target=/var/cache/pacman \
RUN pacman -Syu --noconfirm --needed base base-devel cuda git openmpi hdf5-openmpi cmake

# configure nvidia container runtime
# https://github.com/NVIDIA/nvidia-container-runtime#environment-variables-oci-spec
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV PATH="${PATH}:/opt/cuda/bin"

RUN git clone --branch develop https://github.com/fizban007/CoffeeGPU.git && \
    cd CoffeeGPU && \
    mkdir build && \
    cd build && \
    CXX=/opt/cuda/bin/g++ cmake .. && \
    make
