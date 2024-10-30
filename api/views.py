from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import action as restful_action

from .models import Patient

from .serializers import PatientSerializer


class PatientViewSet(ModelViewSet):

    def get_queryset(self):
        return Patient.objects.all()

    def get_serializer_class(self):
        return PatientSerializer

    def create(self, request, *args, **kwargs):

        serializer = self.get_serializer(data=request.data)

        if serializer.is_valid():
            serializer.save()

            response = {
                "created": True,
                "data": serializer.data,
            }
            return Response(response, status=status.HTTP_201_CREATED)

        else:

            response = {
                "created": False,
                "errors": serializer.errors,
            }
            return Response(response, status=status.HTTP_400_BAD_REQUEST)

    @restful_action(detail=True, methods=["GET"], url_path="get-predictions")
    def get_predictions(self, request, *args, **kwargs):

        response = {
            "message": "all the predictions",
        }

        return Response(response, status=status.HTTP_200_OK)

    @restful_action(detail=True, methods=["GET"], url_path="generate-treatment-plan")
    def generate_treatment_plan(self, request, *args, **kwargs):

        response = {
            "message": "Treatment plan generated successfully.",
        }

        return Response(response, status=status.HTTP_200_OK)
