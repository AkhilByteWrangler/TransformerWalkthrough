#!/bin/bash

# AWS EC2 Deployment Script for Transformer Viz
# Run this on your EC2 instance after initial setup

set -e

echo "Starting deployment..."

# Update system
sudo apt update
sudo apt upgrade -y

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install nginx
sudo apt install -y nginx

# Install PM2 globally
sudo npm install -g pm2

# Clone or pull latest code
if [ -d "transformer-viz" ]; then
    cd transformer-viz
    git pull
else
    git clone YOUR_REPO_URL transformer-viz
    cd transformer-viz
fi

# Install dependencies
npm install
cd client && npm install && cd ..

# Build client
npm run build

# Create .env file if doesn't exist
if [ ! -f .env ]; then
    echo "PORT=5000" > .env
    echo "NODE_ENV=production" >> .env
fi

# Build TypeScript
npx tsc server/index.ts --outDir dist --esModuleInterop --resolveJsonModule --skipLibCheck

# Start with PM2
pm2 delete transformer-viz 2>/dev/null || true
pm2 start dist/server/index.js --name transformer-viz
pm2 save
pm2 startup

echo "Deployment complete!"
echo "Run 'sudo nano /etc/nginx/sites-available/transformer-viz' to configure nginx"
