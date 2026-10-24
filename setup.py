from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="log-sentinel",
    version="1.0.0",
    description="Log Analyzer with Machine Learning — anomaly detection, classification, and pattern recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LogSentinel Team",
    url="https://github.com/janderik/log-sentinel",
    # The code imports itself as "src.*", so the package is installed under that name.
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.3",
        "numpy>=1.26",
        "pandas>=2.1",
        "joblib>=1.3",
        "fastapi>=0.109",
        "uvicorn>=0.27",
        "pydantic>=2.5",
        "click>=8.1",
        "rich>=13.7",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=4.1",
            "mypy>=1.8",
            "ruff>=0.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "logsentinel=src.cli:cli",
            "logsentinelserve=src.api.serve_app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: System :: Monitoring",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
