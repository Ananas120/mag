FROM tensorflow/tensorflow:latest-gpu-notebook

WORKDIR /tensorflow/app

RUN apt-get update && apt-get install -y ffmpeg poppler-utils

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN apt install -y graphviz

COPY . .

CMD ["jupyter", "notebook", "--ip 0.0.0.0", "--no-browser", "--allow-root"]