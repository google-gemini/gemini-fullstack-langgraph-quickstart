#!/usr/bin/env python3
"""
Setup script for Gemini Research Agent MCP Server

This setup script allows for proper installation and distribution of the
MCP server package following Python packaging best practices.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Gemini Research Agent MCP Server"

# Read requirements from requirements.txt
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

setup(
    name="gemini-research-agent-mcp",
    version="1.0.0",
    author="Open Source Project",
    author_email="contact@example.com",
    description="A comprehensive MCP server for AI-powered research using Google Gemini models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/gemini-research-agent-mcp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.7.0",
            "pytest-cov>=4.1.0",
        ],
        "performance": [
            "uvloop>=0.19.0; sys_platform != 'win32'",
        ],
    },
    entry_points={
        "console_scripts": [
            "gemini-research-mcp=server:main",
        ],
    },
    keywords=[
        "mcp",
        "model-context-protocol",
        "gemini",
        "research",
        "ai",
        "google",
        "langchain",
        "web-search",
        "citations",
        "async",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/gemini-research-agent-mcp/issues",
        "Source": "https://github.com/your-username/gemini-research-agent-mcp",
        "Documentation": "https://github.com/your-username/gemini-research-agent-mcp#readme",
    },
    include_package_data=True,
    zip_safe=False,
) 