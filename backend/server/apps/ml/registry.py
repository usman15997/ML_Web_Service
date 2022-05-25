# file backend/server/apps/ml/registry.py
from backend.server.apps.endpoints.models import Endpoint
from backend.server.apps.endpoints.models import MLAlgorithm
from backend.server.apps.endpoints.models import MLAlgorithmStatus

"""
We have the ML code ready and tested. We need to connect it with the server code. For this, I will create the ML registry object, that will keep information about available algorithms and corresponding endpoints.
"""

class MLRegistry:
    def __init__(self):
        self.endpoints = {}

    def add_algorithm(self, endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, owner,
                    algorithm_description, algorithm_code):
        # get endpoint
        endpoint, _ = Endpoint.objects.get_or_create(name=endpoint_name, owner=owner)

        # get algorithm
        database_object, algorithm_created = MLAlgorithm.objects.get_or_create(
                name=algorithm_name,
                description=algorithm_description,
                code=algorithm_code,
                version=algorithm_version,
                owner=owner,
                parent_endpoint=endpoint)
        if algorithm_created:
            status = MLAlgorithmStatus(status = algorithm_status,
                                        created_by = owner,
                                        parent_mlalgorithm = database_object,
                                        active = True)
            status.save()

        # add to registry
        self.endpoints[database_object.id] = algorithm_object