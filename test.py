#!/bin/bash
# Quick setup script for VerilogEval on Linux
# This script automates the entire setup process

set -e

echo "=========================================="
echo "VerilogEval Linux Setup Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux systems"
    exit 1
fi

# Check for required commands
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed"
        return 1
    else
        print_status "$1 is available"
        return 0
    fi
}

print_status "Checking system requirements..."

# Check for essential tools
MISSING_TOOLS=0

if ! check_command conda; then
    print_warning "conda not found. Please install Miniconda or Anaconda"
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    MISSING_TOOLS=1
fi

if ! check_command git; then
    print_warning "git not found. Install with: sudo apt install git"
    MISSING_TOOLS=1
fi

# Check for Ollama
if ! check_command ollama; then
    print_warning "Ollama not found. Will attempt to install..."
    
    read -p "Install Ollama automatically? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        
        # Start Ollama service
        print_status "Starting Ollama service..."
        if command -v systemctl &> /dev/null; then
            sudo systemctl start ollama
            sudo systemctl enable ollama
        else
            print_warning "systemctl not available. You may need to start Ollama manually with 'ollama serve'"
        fi
    else
        print_error "Ollama is required. Please install it manually and re-run this script"
        exit 1
    fi
fi

# Check for iverilog
if ! check_command iverilog; then
    print_warning "iverilog not found. Will attempt to install..."
    
    if command -v apt &> /dev/null; then
        sudo apt update
        sudo apt install -y iverilog
    elif command -v yum &> /dev/null; then
        sudo yum install -y iverilog
    elif command -v pacman &> /dev/null; then
        sudo pacman -S iverilog
    else
        print_warning "Could not auto-install iverilog. Please install manually"
    fi
fi

if [ $MISSING_TOOLS -eq 1 ]; then
    print_error "Please install missing tools and re-run this script"
    exit 1
fi

print_status "All required tools are available!"

# Check if we're in the verilog-eval directory
if [ ! -f "scripts/sv-generate" ]; then
    print_error "Please run this script from the verilog-eval directory"
    print_error "Current directory: $(pwd)"
    exit 1
fi

# Create conda environment
print_status "Creating conda environment 'verilogeval'..."

if conda env list | grep -q "verilogeval"; then
    print_warning "Environment 'verilogeval' already exists"
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n verilogeval
    else
        print_status "Using existing environment"
    fi
fi

if [ -f "environment.yml" ]; then
    print_status "Using environment.yml..."
    conda env create -f environment.yml
else
    print_status "Creating environment manually..."
    conda create -n verilogeval python=3.10 -y
    
    # Activate environment and install packages
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate verilogeval
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing packages from requirements.txt..."
        pip install -r requirements.txt
    else
        print_status "Installing packages manually..."
        pip install langchain==0.3.27 langchain-core==0.3.75 langchain-community==0.3.29 \
            langchain-ollama==0.3.7 langchain-openai==0.3.32 numpy requests pydantic \
            tenacity tqdm python-dotenv PyYAML ollama
    fi
fi

# Activate the environment for testing
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verilogeval

print_status "Testing installation..."

# Test Python imports
python -c "
try:
    from langchain_ollama import OllamaLLM
    print('✓ LangChain Ollama import successful')
except ImportError as e:
    print(f'✗ LangChain Ollama import failed: {e}')
    exit(1)

try:
    import numpy
    print('✓ NumPy import successful')
except ImportError as e:
    print(f'✗ NumPy import failed: {e}')
"

# Test Ollama connection
print_status "Testing Ollama connection..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    print_warning "Ollama server not responding. Starting Ollama..."
    
    # Try to start Ollama
    if command -v systemctl &> /dev/null; then
        sudo systemctl start ollama
        sleep 3
    else
        print_warning "Starting Ollama manually (this will run in background)"
        ollama serve &
        sleep 5
    fi
fi

# Check what models are available
print_status "Available Ollama models:"
ollama list

# Suggest models to pull if none are available
if ! ollama list | grep -q "NAME"; then
    print_warning "No models found. Recommended models to pull:"
    echo "  ollama pull llama3.2:1b      # Smallest, fastest (1.3GB)"
    echo "  ollama pull mistral:latest   # Good general performance (4.1GB)" 
    echo "  ollama pull codellama:7b     # Good for code generation (3.8GB)"
    echo ""
    read -p "Pull llama3.2:1b model now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Pulling llama3.2:1b model..."
        ollama pull llama3.2:1b
    fi
fi

# Test iverilog
print_status "Testing iverilog..."
iverilog -V

# Make scripts executable
print_status "Setting up executable permissions..."
chmod +x scripts/sv-generate
if [ -f "simple_eval.py" ]; then
    chmod +x simple_eval.py
fi
if [ -f "analyze_results.py" ]; then
    chmod +x analyze_results.py
fi

# Create activation helper script
print_status "Creating activation helper script..."
cat > activate_verilogeval.sh << 'EOF'
#!/bin/bash
# Source this script to activate the VerilogEval environment
# Usage: source activate_verilogeval.sh

source $(conda info --base)/etc/profile.d/conda.sh
conda activate verilogeval

echo "VerilogEval environment activated!"
echo "Available commands:"
echo "  python simple_eval.py --help"
echo "  python scripts/sv-generate --list-models" 
echo "  ollama list"

# Set up environment variables
export VERILOGEVAL_ROOT=$(pwd)
export PYTHONPATH="${VERILOGEVAL_ROOT}:${PYTHONPATH}"
EOF

chmod +x activate_verilogeval.sh

print_status "Setup complete!"
echo ""
echo "=========================================="
echo "Next steps:"
echo "=========================================="
echo "1. Activate environment: source activate_verilogeval.sh"
echo "2. Test the setup: python simple_eval.py --model llama3.2:1b --problems 1 --verbose"
echo "3. Run evaluation: python simple_eval.py --model mistral:latest --problems 5"
echo "4. Analyze results: python analyze_results.py results/model_name"
echo ""
echo "If you don't have the modified scripts from Windows, you'll need to:"
echo "- Copy the modified scripts/sv-generate file"
echo "- Copy simple_eval.py and analyze_results.py" 
echo "- Or re-apply the Ollama modifications manually"
echo ""
print_status "Setup script completed successfully!"
