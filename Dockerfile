FROM python:3.10-slim

WORKDIR /app/

# Копируем только requirements.txt для установки зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаем необходимые директории и устанавливаем SSH сервер
RUN mkdir -p /app/dataset_parts /app/models /var/run/sshd && \
    apt update && apt install -y --no-install-recommends openssh-server && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    apt clean && rm -rf /var/lib/apt/lists/*

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]

# Копируем весь остальной код
COPY . .
