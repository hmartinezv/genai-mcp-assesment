#!/usr/bin/env bash
set -euo pipefail

APP_NAME="support-chatbot"
IMAGE_NAME="support-chatbot:latest"
CONTAINER_NAME="support-chatbot-app"
PORT_INTERNAL="8501"

# Change to "direct" if you want :8501 public without nginx
MODE="${1:-nginx}"  # nginx | direct

if [[ ! -f ".env" ]]; then
  echo "ERROR: .env file not found in repo root."
  echo "Create .env with GROQ_API_KEY and MCP_SERVER_URL."
  exit 1
fi

echo "==> Checking Docker..."
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found. Installing (Ubuntu/Debian)..."
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER" || true
  echo "Docker installed. You may need to log out/in for group changes to apply."
fi

echo "==> Building Docker image..."
docker build -t "$IMAGE_NAME" .

echo "==> Stopping old container (if exists)..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

if [[ "$MODE" == "direct" ]]; then
  echo "==> Running container (public on :8501)..."
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --env-file .env \
    -p 8501:8501 \
    "$IMAGE_NAME"

  # Open firewall on VM if ufw exists
  if command -v ufw >/dev/null 2>&1; then
    sudo ufw allow 8501/tcp || true
  fi

  echo "==> Done."
  echo "Public URL: http://<YOUR_VM_EXTERNAL_IP>:8501"
  exit 0
fi

echo "==> Running container (internal only; nginx will expose port 80)..."
# bind only to localhost so nginx is the public entrypoint
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --env-file .env \
  -p 127.0.0.1:8501:8501 \
  "$IMAGE_NAME"

echo "==> Installing nginx (if missing)..."
if ! command -v nginx >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y nginx
fi

echo "==> Configuring nginx reverse proxy..."
NGINX_SITE="/etc/nginx/sites-available/${APP_NAME}"
sudo tee "$NGINX_SITE" >/dev/null <<'EOF'
server {
  listen 80;
  server_name _;

  client_max_body_size 10m;

  location / {
    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;

    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Websocket support for Streamlit
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
  }
}
EOF

sudo ln -sf "$NGINX_SITE" "/etc/nginx/sites-enabled/${APP_NAME}"
sudo rm -f /etc/nginx/sites-enabled/default || true

sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx

# Open firewall on VM if ufw exists
if command -v ufw >/dev/null 2>&1; then
  sudo ufw allow 80/tcp || true
fi

echo "==> Done."
echo "Public URL: http://<YOUR_VM_EXTERNAL_IP>/"
echo ""
echo "IMPORTANT: In GCP Firewall rules, allow inbound TCP 80 (and/or 8501 if using direct mode)."