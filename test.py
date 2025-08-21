# Stop the Ollama service
sudo systemctl stop ollama

# Remove the Ollama service
sudo systemctl disable ollama
sudo rm /etc/systemd/system/ollama.service

# Remove the Ollama binary
sudo rm $(which ollama)

# Remove Ollama user and group (if they exist)
sudo userdel ollama
sudo groupdel ollama
