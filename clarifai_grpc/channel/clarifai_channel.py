import os

import requests

from clarifai_grpc.channel.grpc_json_channel import GRPCJSONChannel


RETRIES = 2  # if connections fail retry a couple times.
CONNECTIONS = 20  # number of connections to maintain in pool.


class ClarifaiChannel:
  @classmethod
  def get_json_channel(
      cls,
      api_key=os.environ.get('CLARIFAI_API_KEY', 'no_key'),
      base_url=os.environ.get('CLARIFAI_API_BASE', 'https://api.clarifai.com')
  ):
    session = cls._make_requests_session()

    return GRPCJSONChannel(session=session, key=api_key, base_url=base_url)

  @staticmethod
  def _make_requests_session():
    http_adapter = requests.adapters.HTTPAdapter(
      max_retries=RETRIES, pool_connections=CONNECTIONS, pool_maxsize=CONNECTIONS)

    session = requests.Session()
    session.mount('http://', http_adapter)
    session.mount('https://', http_adapter)
    return session
