from setuptools import find_packages, setup

with open("requirements-dev.txt") as f:
    test_requirements = f.read().splitlines()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


extra_requirements = {"dev": test_requirements}

setup_requirements = ["pytest-runner"]

setup(
    name="pandas-transformers",
    packages=find_packages("src"),
    package_dir={"": "src"},
    version="0.1.0",
    description="Pandas equivalents of (e.g.) the beloved scikit-learn transformers.",
    author="Nedim Bayrakdar",
    license="MIT",
    test_suite="tests",
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require=extra_requirements,
    setup_requires=setup_requirements,
)
