# Contributing to My Early Days of Machine Learning

This repository documents my learning journey from 2018. While the original code is preserved as-is for historical purposes, we welcome improvements to documentation, README, and educational value.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/My-early-days-of-machine-learning.git
   cd My-early-days-of-machine-learning
   ```

3. **Set up development environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make your changes** (documentation, README improvements, etc.)

6. **Submit a pull request**

## 🔧 Development Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook / JupyterLab
- Git

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install flake8 black isort  # Development tools
```

### Running Notebooks
```bash
jupyter notebook
```

## 📝 Code Style

We follow Python best practices for any new code:

### Formatting
- **Black** for code formatting
- **isort** for import sorting
- Line length: 88 characters (Black default)

```bash
# Format all Python files
black .

# Sort imports
isort .
```

### Linting
- **flake8** for style checking

```bash
flake8 . --max-line-length=88 --extend-ignore=E203,W503 --exclude=.git,__pycache__,.venv,venv,env,ML_Notes,Gradient_descent_notes,monte_carlo
```

## 🎯 Contribution Guidelines

### What We Welcome
- **README improvements**: Better explanations, clearer structure
- **Documentation**: Adding context, explanations, learning notes
- **Code comments**: Explaining the "why" behind beginner approaches
- **Educational enhancements**: Comparisons with modern practices
- **Typo fixes**: In markdown cells and documentation

### What We Don't Change
- **Original 2018 notebook code**: Preserved as historical artifacts
- **Hardcoded paths**: Part of the learning journey
- **Beginner mistakes**: They tell the story

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Changes are focused on documentation/educational value
- [ ] Original notebook code is not modified
- [ ] README.md is updated if needed
- [ ] Changes improve clarity for learners

### Pull Request Description
Include:
- **Purpose**: What does this PR accomplish?
- **Changes**: What specific changes were made?
- **Educational Value**: How does this help learners?

### Example PR Description
```
## Purpose
Add "Then vs Now" comparison table to README showing how approaches have evolved

## Changes
- Added table comparing 2018 manual implementations vs modern scikit-learn
- Added explanation of vectorization benefits
- Updated notebook list with learning objectives

## Educational Value
Helps beginners understand both the fundamentals and modern best practices
```

## 🐛 Bug Reports

When reporting issues, please include:

1. **Environment**: Python version, OS, package versions
2. **Description**: Clear description of the issue
3. **Location**: Which notebook/file has the issue
4. **Expected vs Actual**: What you expected vs what happened

## 💡 Feature Requests

For improvements:
1. **Check existing issues** to avoid duplicates
2. **Describe the educational value** - how does this help learners?
3. **Propose implementation** if you have ideas
4. **Keep historical integrity** - original code stays unchanged

## 📚 Documentation

### README Documentation
- Use clear, descriptive sections
- Include learning objectives for each notebook
- Add "Then vs Now" comparisons
- Reference modern equivalents

### Adding Documentation
- Update README.md for user-facing changes
- Add markdown cells to notebooks for context (not code changes)
- Include references to learning resources

## 🤝 Community Guidelines

- **Be respectful** and inclusive
- **Help others** learn and contribute
- **Ask questions** if something is unclear
- **Provide constructive feedback**

## 📞 Getting Help

- **GitHub Issues**: For documentation issues and suggestions
- **GitHub Discussions**: For questions about ML concepts

Thank you for contributing to this learning journey! 🎉