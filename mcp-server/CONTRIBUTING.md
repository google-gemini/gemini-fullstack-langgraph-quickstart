# Contributing to Gemini Research Agent MCP Server

Thank you for your interest in contributing to the Gemini Research Agent MCP Server! We welcome contributions from the community and are excited to see what you'll build.

## 🤝 Ways to Contribute

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new functionality or improvements
- **Code Contributions**: Submit bug fixes, new features, or optimizations
- **Documentation**: Improve docs, examples, or tutorials
- **Testing**: Help improve test coverage and quality
- **Community**: Help others in discussions and issues

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Git for version control

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/gemini-research-agent-mcp.git
   cd gemini-research-agent-mcp
   ```

2. **Set Up Environment**
   ```bash
   make setup
   cp env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

3. **Install Dependencies**
   ```bash
   make install-dev
   ```

4. **Run Tests**
   ```bash
   make test
   ```

5. **Start the Server**
   ```bash
   make run
   ```

## 📝 Development Guidelines

### Code Style

We follow Python best practices and maintain high code quality:

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type hints for all functions and methods
- **Docstrings**: Write comprehensive docstrings for all public functions
- **Line Length**: Maximum 100 characters per line

**Formatting Tools:**
```bash
make format          # Auto-format with black
make lint            # Check with flake8
make type-check      # Validate with mypy
```

### Testing

- **Write Tests**: All new features must include tests
- **Test Coverage**: Maintain >90% test coverage
- **Test Types**: Include unit, integration, and error condition tests

```bash
make test            # Run all tests
make test-coverage   # Run tests with coverage report
```

### Commit Guidelines

We follow conventional commit messages:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(research): add caching for search results
fix(citations): handle malformed URLs gracefully
docs(readme): update installation instructions
test(server): add tests for error handling
```

## 🔄 Pull Request Process

### Before Submitting

1. **Check Existing Issues**: Look for related issues or discussions
2. **Create Issue**: For significant changes, create an issue first
3. **Branch**: Create a feature branch from `main`
4. **Code**: Implement your changes following our guidelines
5. **Test**: Ensure all tests pass and add new tests if needed
6. **Document**: Update documentation as needed

### Submitting

1. **Quality Checks**
   ```bash
   make check-all       # Run all quality checks
   ```

2. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat(scope): your descriptive message"
   git push origin your-feature-branch
   ```

3. **Create Pull Request**
   - Use a descriptive title
   - Fill out the PR template
   - Link related issues
   - Request review from maintainers

### PR Requirements

- ✅ All tests pass
- ✅ Code follows style guidelines
- ✅ Documentation updated (if applicable)
- ✅ Changelog entry added (for significant changes)
- ✅ No merge conflicts with main branch

## 🐛 Bug Reports

When reporting bugs, please include:

- **Environment**: OS, Python version, package versions
- **Description**: Clear description of the issue
- **Reproduction**: Steps to reproduce the problem
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Logs**: Relevant error messages or logs
- **Additional Context**: Screenshots, configuration files, etc.

**Bug Report Template:**
```markdown
## Bug Description
Brief description of the issue

## Environment
- OS: [e.g., macOS 14.0]
- Python: [e.g., 3.11.5]
- MCP Server Version: [e.g., 1.0.0]

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Error Logs
```
[paste error logs here]
```

## Additional Context
Any other relevant information
```

## 💡 Feature Requests

For feature requests, please provide:

- **Use Case**: Why is this feature needed?
- **Description**: Detailed description of the proposed feature
- **Alternatives**: Any alternative solutions considered
- **Implementation**: If you have ideas about implementation

## 🏗 Architecture Guidelines

### Code Organization

- **Modularity**: Keep functions focused and reusable
- **Separation of Concerns**: Separate business logic from infrastructure
- **Error Handling**: Comprehensive error handling with meaningful messages
- **Logging**: Structured logging for debugging and monitoring

### MCP Compliance

- Follow MCP protocol specifications
- Ensure compatibility with standard MCP clients
- Maintain backward compatibility when possible
- Document any protocol extensions

### Performance

- **Async/Await**: Use async patterns for I/O operations
- **Resource Management**: Proper cleanup of resources
- **Caching**: Implement appropriate caching strategies
- **Rate Limiting**: Respect API rate limits

## 📚 Documentation

### Types of Documentation

- **API Documentation**: Function and method documentation
- **User Guides**: How-to guides for users
- **Developer Docs**: Architecture and development guides
- **Examples**: Working examples and tutorials

### Writing Guidelines

- **Clear and Concise**: Use simple, clear language
- **Examples**: Include practical examples
- **Up-to-Date**: Keep documentation current with code changes
- **Accessible**: Consider accessibility in documentation

## 🔒 Security

### Reporting Security Issues

For security vulnerabilities, please:

1. **Do NOT** create a public issue
2. Email security concerns to [security@project.com]
3. Include detailed information about the vulnerability
4. Allow time for assessment and resolution

### Security Guidelines

- **API Keys**: Never commit API keys or secrets
- **Input Validation**: Validate all inputs
- **Dependencies**: Keep dependencies updated
- **Access Control**: Implement proper access controls

## 🌟 Recognition

Contributors will be recognized in:

- **README.md**: Contributors section
- **CHANGELOG.md**: Release notes
- **GitHub**: Contributor graphs and statistics

## 📞 Getting Help

If you need help:

- **Documentation**: Check existing documentation
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Community**: Join our community channels

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You

Thank you for contributing to the Gemini Research Agent MCP Server! Your contributions help make this project better for everyone.

---

*This document is based on best practices from the open source community and is regularly updated to reflect our evolving processes.* 