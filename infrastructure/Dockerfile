FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update &&  \
	apt install -y  \
	openssh-server \
	libgl1 \
	tmux \
	byobu

ENV HOME /home/developer
RUN mkdir -p $HOME/infrastructure
WORKDIR $HOME/developer
ENV CSV=${CSV:-$HOME/developer/data/data.csv}

COPY . .
COPY ssh/sshd_config /etc/ssh/sshd_config
EXPOSE 22
EXPOSE 8888

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN rm -rf ssh

RUN useradd -rm -d $HOME -s /bin/bash -g root -G sudo -u 1000 developer
RUN echo 'developer:password' | chpasswd

SHELL ["/bin/bash", "-l", "-c"]

ENTRYPOINT service ssh start && jupyter-lab \
	--ip 0.0.0.0 \
	--port 8888 \
	--no-browser \
	--NotebookApp.notebook_dir='/home/developer' \
	--ServerApp.terminado_settings="shell_command=['/bin/bash']" \
	--allow-root