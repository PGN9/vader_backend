#!/usr/bin/env bash
set -e

echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing requirements..."
pip install -r requirements.txt
