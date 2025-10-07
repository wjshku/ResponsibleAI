#!/bin/bash
# Setup script for Car Damage Inpainting with uv

echo "🚗 Setting up Car Damage Inpainting with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv is installed"

# Create virtual environment
echo "📦 Creating virtual environment..."
uv venv

# Activate virtual environment and install dependencies
echo "📥 Installing dependencies..."
uv pip install -e .

echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "   source .venv/bin/activate"
echo ""
echo "To run the inpainting system:"
echo "   python inpaint.py --use-local --num-samples 4"
echo "   python inpaint.py --api-key YOUR_KEY --num-samples 4"
echo ""
echo "To run tests:"
echo "   python test.py"
