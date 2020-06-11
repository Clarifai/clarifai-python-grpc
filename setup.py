import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clarifai-grpc",
    version="6.5.0",
    author="Clarifai",
    author_email="support@clarifai.com",
    description="Clarifai gRPC API Client",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Clarifai/clarifai-python-grpc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
    python_requires='>=3.6',
    install_requires=[
        "grpcio>=1.25.0",
        "protobuf>=3.10.0",
        "googleapis-common-protos>=1.6.0",
        "requests>=2.22.0",
    ]
)
