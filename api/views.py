from rest_framework.viewsets import ModelViewSet


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
