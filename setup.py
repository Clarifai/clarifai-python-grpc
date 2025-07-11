import re

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages(include=["clarifai_grpc*"])

# Load the version
with open("./clarifai_grpc/__init__.py") as f:
    content = f.read()
_search_version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content)
assert _search_version
version = _search_version.group(1)

setuptools.setup(
    name="clarifai-grpc",
    version=f"{version}",
    author="Clarifai",
    author_email="support@clarifai.com",
    description="Clarifai gRPC API Client",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Clarifai/clarifai-python-grpc",
    packages=packages,
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
    python_requires='>=3.8',
    install_requires=[
        "grpcio>=1.53.2 ; python_version < '3.13'",
        "grpcio>=1.68.0 ; python_version >= '3.13'",
        "protobuf>=5.29.5",
        "googleapis-common-protos>=1.57.0",
    ],
    package_data={p: ["*.pyi"] for p in packages},
    include_package_data=True,
)
