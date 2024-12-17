#!/bin/bash

read -p "SSH Port: " port
read -p "SSH IP: " IP

NEW_IP=${IP}
NEW_PORT=${port}
CONFIG_FILE="$HOME/.ssh/config_runpod"

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found."
    exit 1
fi

sed -i '' "s/^ *HostName .*/  HostName $NEW_IP/" "$CONFIG_FILE"
sed -i '' "s/^ *Port .*/  Port $NEW_PORT/" "$CONFIG_FILE"


echo "Updated $CONFIG_FILE with HostName=$NEW_IP and Port=$NEW_PORT."


scp -P ${port} ~/Dropbox/Stevens\ Courses/AAI-595A\ Applied\ Machine\ Learning/Final\ Project/repo/runpod/setup_runpod.sh root@${IP}:/root/setup_runpod.sh
ssh -p ${port} -t root@${IP} 'chmod +x /root/setup_runpod.sh && ./setup_runpod.sh'
