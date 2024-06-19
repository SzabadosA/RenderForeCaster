FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "rfc_forecasting_env", "/bin/bash", "-c"]

COPY . .

CMD ["conda", "run", "--no-capture-output", "-n", "rfc_forecasting_env", "python", "app.py"]
