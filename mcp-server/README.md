# 🔬 Gemini Research Agent MCP Server

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![MCP](https://img.shields.io/badge/MCP-1.0%2B-green.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.5--flash--preview--05--20-orange.svg)
![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)

[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Security Policy](https://img.shields.io/badge/security-policy-important.svg)](SECURITY.md)
[![Issues](https://img.shields.io/github/issues/your-username/gemini-research-agent-mcp.svg)](https://github.com/your-username/gemini-research-agent-mcp/issues)
[![Stars](https://img.shields.io/github/stars/your-username/gemini-research-agent-mcp.svg?style=social)](https://github.com/your-username/gemini-research-agent-mcp/stargazers)

**🚀 Production-ready MCP server for advanced AI-powered research with configurable effort levels**

[Getting Started](#-quick-start) •
[Documentation](#-usage) •
[Contributing](CONTRIBUTING.md) •
[Changelog](CHANGELOG.md) •
[Security](SECURITY.md)

</div>

A comprehensive Model Context Protocol (MCP) server that provides advanced research capabilities using Google's Gemini AI models. This server offers tiered effort levels, intelligent web research, citation tracking, and comprehensive answer synthesis.

## 🚀 Features

- **Multi-tier Research**: Three configurable effort levels (low, medium, high) with different search limits
- **Intelligent Query Generation**: AI-powered search query creation and optimization
- **Iterative Research Loops**: Multiple research cycles with reflection and gap analysis
- **Citation Tracking**: Comprehensive source validation and citation management
- **Robust Error Handling**: Graceful degradation with fallback mechanisms
- **Async Performance**: Optimized for concurrent operations and high throughput
- **Industry Standards**: Clean code, comprehensive logging, and proper documentation

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Effort Levels](#-effort-levels)
- [Architecture](#-architecture)
- [Contributing](#-contributing)
- [License](#-license)
- [Support](#-support)

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- Google Cloud API key with Gemini access
- Model Context Protocol (MCP) client

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd mcp-new

# Install dependencies
pip install -r requirements.txt

# Or install with development dependencies
pip install -r requirements.txt
pip install -e .
```

### Environment Setup

Create a `.env` file in the project root:

```env
# Required: Google Gemini API Key
GEMINI_API_KEY=your_api_key_here

# Optional: Logging level
LOG_LEVEL=INFO
```

## 🚀 Quick Start

### Basic Usage

```python
# Start the MCP server
python server.py
```

### Using with MCP Clients

The server provides three main tools:

1. **research_topic**: Conduct comprehensive research
2. **get_effort_levels**: Get information about effort tiers
3. **get_server_status**: Check server status and active sessions

### Example Research

```json
{
  "tool": "research_topic",
  "arguments": {
    "topic": "artificial intelligence trends 2024",
    "effort": "medium"
  }
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | - | Google Gemini API key |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Model Configuration

The server uses multiple Gemini models for different tasks:

- **Research Model**: `gemini-2.5-flash-preview-05-20` (Primary research)
- **Query Model**: `gemini-2.0-flash` (Query generation)
- **Reflection Model**: `gemini-2.5-flash-preview-04-17` (Research reflection)
- **Answer Model**: `gemini-2.5-pro-preview-05-06` (Final answer synthesis)

## 📖 Usage

### MCP Tools

#### research_topic(topic, effort="medium")

Conduct comprehensive research on any topic with configurable effort levels.

**Parameters:**
- `topic` (str): The research topic or question to investigate
- `effort` (str): Research effort level - "low", "medium", or "high"

**Returns:** Comprehensive research report with citations and sources

**Example:**
```python
result = await research_topic(
    topic="climate change impacts on agriculture",
    effort="high"
)
```

#### get_effort_levels()

Get detailed information about available research effort levels.

**Returns:** Markdown-formatted breakdown of effort levels

#### get_server_status()

Get current server status and active research sessions.

**Returns:** Server status and configuration information

### MCP Resources

#### research://documentation

Access comprehensive documentation for the server.

**URI:** `research://documentation`

## 🎯 Effort Levels

| Level | Max Searches | Max Loops | Initial Queries | Best For |
|-------|-------------|-----------|-----------------|----------|
| **Low** | 10 | 1 | 2 | Quick facts, simple questions, time-sensitive research |
| **Medium** | 100 | 3 | 4 | General research needs, balanced depth and speed |
| **High** | 1000 | 5 | 6 | Complex topics, academic research, comprehensive analysis |

### Effort Level Details

- **Low Effort**: Optimized for quick answers and fact-checking
- **Medium Effort**: Balanced approach for most research needs
- **High Effort**: Comprehensive research for complex topics requiring deep analysis

## 🏗 Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │───▶│   FastMCP Server │───▶│  Gemini Models  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ Research Engine  │
                       │ • Query Gen      │
                       │ • Web Search     │
                       │ • Reflection     │
                       │ • Synthesis      │
                       └──────────────────┘
```

### Key Features

- **Async Architecture**: Built with asyncio for high performance
- **State Management**: Tracks active research sessions with cleanup
- **Error Resilience**: Comprehensive error handling with fallbacks
- **Citation System**: Advanced URL processing and source validation
- **Logging**: Structured logging for debugging and monitoring

### Research Workflow

1. **Query Generation**: AI generates diverse, targeted search queries
2. **Initial Research**: Conducts parallel web searches
3. **Reflection Loop**: Analyzes results and identifies knowledge gaps
4. **Follow-up Research**: Generates additional queries based on gaps
5. **Synthesis**: Combines all findings into comprehensive answer

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd mcp-new

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest

# Format code
black .
flake8 .
mypy .
```

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Add unit tests for new features
- Maintain >90% test coverage

### Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help

- **Documentation**: Check the comprehensive docs in the server
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join community discussions

### Common Issues

#### API Key Issues
```bash
Error: GEMINI_API_KEY environment variable is required
```
**Solution**: Set your Google Gemini API key in the `.env` file

#### Import Errors
```bash
ImportError: FastMCP is required
```
**Solution**: Install MCP framework: `pip install mcp`

#### Model Access Issues
```bash
Error: Model not available
```
**Solution**: Ensure your API key has access to Gemini models

### Performance Tips

- Use appropriate effort levels for your needs
- Monitor active sessions with `get_server_status`
- Consider rate limiting for production deployments
- Use async patterns when integrating with other systems

## 🙏 Acknowledgments

- **Anthropic**: For the Model Context Protocol specification
- **Google**: For the powerful Gemini AI models
- **LangChain**: For the excellent integration framework
- **FastMCP**: For the high-performance MCP server framework

## 📊 Metrics and Monitoring

The server provides built-in metrics:

- Research session tracking
- Search query performance
- Error rates and types
- Resource usage statistics

Access these via the `get_server_status` tool or check the logs for detailed information.

---

**Made with ❤️ for the open-source community**

*For more information, visit our [documentation](research://documentation) resource.* 