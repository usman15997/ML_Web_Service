from django.shortcuts import render
from rest_framework import viewsets
from rest_framework import mixins
from django.db import transaction
from rest_framework.exceptions import APIException
from backend.server.apps.endpoints.models import Endpoint
from backend.server.apps.endpoints.serializers import EndpointSerializer

from backend.server.apps.endpoints.models import MLAlgorithm
from backend.server.apps.endpoints.serializers import MLAlgorithmSerializer

from backend.server.apps.endpoints.models import MLAlgorithmStatus
from backend.server.apps.endpoints.serializers import MLAlgorithmStatusSerializer

from backend.server.apps.endpoints.models import MLRequest
from backend.server.apps.endpoints.serializers import MLRequestSerializer


"""
For each model, we created a view which will allow to retrieve single object or list of objects. We will not allow to create or modify Endpoints, MLAlgorithms by REST API. The code to to handle creation of new ML related objects will be on server side, I will describe it in the next chapter.

We will allow to create MLAlgorithmStatus objects by REST API. We don’t allow to edit statuses for ML algorithms as we want to keep all status history.

We allow to edit MLRequest objects, however only feedback field (please take a look at serializer definition).
"""

class EndpointViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    serializer_class = EndpointSerializer
    queryset = Endpoint.objects.all()


class MLAlgorithmViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    serializer_class = MLAlgorithmSerializer
    queryset = MLAlgorithm.objects.all()


def deactivate_other_statuses(instance):
    old_statuses = MLAlgorithmStatus.objects.filter(parent_mlalgorithm = instance.parent_mlalgorithm,
                                                        created_at__lt=instance.created_at,
                                                        active=True)
    for i in range(len(old_statuses)):
        old_statuses[i].active = False
    MLAlgorithmStatus.objects.bulk_update(old_statuses, ["active"])

class MLAlgorithmStatusViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.CreateModelMixin):
    serializer_class = MLAlgorithmStatusSerializer
    queryset = MLAlgorithmStatus.objects.all()
    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance = serializer.save(active=True)
                # set active=False for other statuses
                deactivate_other_statuses(instance)

        except Exception as e:
            raise APIException(str(e))

class MLRequestViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.UpdateModelMixin
):
    serializer_class = MLRequestSerializer
    queryset = MLRequest.objects.all()
