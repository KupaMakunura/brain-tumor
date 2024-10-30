from rest_framework import serializers
from .models import Patient


class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = [
            "id",
            "name",
            "age",
            "sex",
            "health_history",
            "prior_treatments",
            "allergies",
            "existing_conditions",
            "family_history",
            "image",
            "created_at",
            "updated_at",
        ]

        class Meta:
            read_only_fields = ["id", "created_at", "updated_at"]
