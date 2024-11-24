from setuptools import setup, find_packages

setup(
    name="nlp-fake-review-detection",
    version="1.0.0",
    description="NLP-based system for detecting fake product reviews using machine learning",
    author="Shrey",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'nltk',
        'vaderSentiment',
        'joblib',
        'matplotlib',
        'seaborn'
    ],
    python_requires='>=3.8',
)
