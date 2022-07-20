import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages(include=["clarifai_grpc*"])

setuptools.setup(
    name="clarifai-grpc",
    version="8.7.0",
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
    python_requires='>=3.6',
    install_requires=[
        "grpcio>=1.44.0",
        "protobuf>= 3.15.6, <=3.19.3",
        "googleapis-common-protos>=1.53.0",
        "requests>=2.25.1",
    ],
    package_data={p: ["*.pyi"] for p in packages},
    include_package_data=True
)
