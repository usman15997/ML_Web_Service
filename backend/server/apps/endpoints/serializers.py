from rest_framework import serializers
from backend.server.apps.endpoints.models import Endpoint
from backend.server.apps.endpoints.models import MLAlgorithm
from backend.server.apps.endpoints.models import MLAlgorithmStatus
from backend.server.apps.endpoints.models import MLRequest


"""
Serializers will help with packing and unpacking database objects into JSON objects. In Endpoints and MLAlgorithm serializers, we defined all read-only fields. This is because, we will create and modify our objects only on the server-side.For MLAlgorithmStatus, fields status, created_by, created_at and parent_mlalgorithm are in read and write mode, we will use the to set algorithm status by REST API. For MLRequest serializer there is a feedback field that is left in read and write mode - it will be needed to provide feedback about predictions to the server.

The MLAlgorithmSerializer is more complex than others. It has one filed current_status that represents the latest status from MLAlgorithmStatus.
"""

class EndpointSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Endpoint
        read_only_fields = ("id", "name", "owner", "created_at")
        fields = read_only_fields

class MLAlgorithmSerializer(serializers.ModelSerializer):

    current_status = serializers.SerializerMethodField(read_only=True)

    def get_current_status(self, mlalgorithm):
        return MLAlgorithmStatus.objects.filter(parent_mlalgorithm=mlalgorithm).latest('created_at').status

    class Meta:
        model = MLAlgorithm
        read_only_fields = ("id", "name", "description", "code",
                            "version", "owner", "created_at",
                            "parent_endpoint", "current_status")
        fields = read_only_fields

class MLAlgorithmStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLAlgorithmStatus
        read_only_fields = ("id", "active")
        fields = ("id", "active", "status", "created_by", "created_at",
                            "parent_mlalgorithm")

class MLRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLRequest
        read_only_fields = (
            "id",
            "input_data",
            "full_response",
            "response",
            "created_at",
            "parent_mlalgorithm",
        )
        fields =  (
            "id",
            "input_data",
            "full_response",
            "response",
            "feedback",
            "created_at",
            "parent_mlalgorithm",
        )
