# Use the PyTorch base image with CUDA and cuDNN pre-installed
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda only if not already installed
RUN if [ ! -d "/opt/conda" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
        bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
        rm Miniconda3-latest-Linux-x86_64.sh && \
        /opt/conda/bin/conda clean -y --all; \
    fi

# Set path to conda
ENV PATH /opt/conda/bin:$PATH

# Copy environment.yml
COPY environment.yml .

# Create the environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "rfc_ml_trainer_env", "/bin/bash", "-c"]

# Copy the application files
COPY . .

# Install Python dependencies within the Conda environment
RUN conda run -n rfc_ml_trainer_env pip install --upgrade pip && \
    conda run -n rfc_ml_trainer_env pip install -r requirements.txt

# Entry point for the application
CMD ["conda", "run", "--no-capture-output", "-n", "rfc_ml_trainer_env", "python", "train_model.py"]
